"""Redis Streams 三阶段消费者

从 Redis Streams 消费消息，执行文档摄入管线的三个阶段：
  Stage 1: doc_ingest → 解析 + 切片 + 向量化
  Stage 2: graph_extract → LLM 实体抽取
  Stage 3: vector_sync → Neo4j 写入 + 增量聚类 + 摘要更新
"""
import asyncio
import time

from backend.observability import get_logger
from backend.pipeline.stream_queue import (
    get_stream_queue, DOC_INGEST, GRAPH_EXTRACT, VECTOR_SYNC,
    PARSER_GROUP, EXTRACTOR_GROUP, SYNCER_GROUP,
)

log = get_logger("pipeline.stream_consumer")


def handle_doc_ingest(data: dict) -> dict:
    """Stage 1: 解析文档 + 切片 + 向量化。"""
    from backend.documents.loader import DocumentLoader
    from backend.storage.parent_chunk_store import ParentChunkStore
    from backend.milvus.client import MilvusManager
    from backend.milvus.writer import MilvusWriter
    from backend.embedding.service import EmbeddingService
    from backend.documents.fingerprint import compute_file_hash
    from backend.storage.doc_lifecycle import upsert_document_index
    from backend.storage.graph_cleanup import cleanup_by_filename

    filename = data["filename"]
    file_path = data["file_path"]
    file_hash = data.get("file_hash", "")

    log.info("stage1_start", filename=filename)

    try:
        # 1. 清理旧数据
        milvus_mgr = MilvusManager()
        milvus_mgr.init_collection()
        milvus_mgr.delete(f'filename == "{filename}"')
        ParentChunkStore().delete_by_filename(filename)
        cleanup_by_filename(filename)

        # 2. 解析 + 切片
        tenant_id = data.get("tenant_id", 0)
        access_level = data.get("access_level", 0)
        loader = DocumentLoader()
        new_docs = loader.load_document(file_path, filename)
        parent_docs = [d for d in new_docs if int(d.get("chunk_level", 0)) in (1, 2)]
        leaf_docs = [d for d in new_docs if int(d.get("chunk_level", 0)) == 3]

        # 2.1 Inject tenant_id and access_level (mirrors routes.py sync path)
        for doc in parent_docs:
            doc["tenant_id"] = tenant_id
        for doc in leaf_docs:
            doc["tenant_id"] = tenant_id
            doc["access_level"] = access_level

        # 3. 存储父块
        parent_store = ParentChunkStore()
        parent_store.upsert_documents(parent_docs)

        # 4. 向量化写入 Milvus
        embedding_service = EmbeddingService()
        writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_mgr)
        writer.write_documents(leaf_docs)

        # 5. 更新文档索引
        if not file_hash:
            file_hash = compute_file_hash(file_path)
        upsert_document_index(filename, file_hash, len(leaf_docs), tenant_id=tenant_id)

        log.info("stage1_complete", filename=filename, chunks=len(leaf_docs))

        # 传递 L1 块文本给 Stage 2（用于图谱抽取，语义更完整）
        l1_chunks = [d for d in new_docs if int(d.get("chunk_level", 0)) == 1]
        if not l1_chunks:
            l1_chunks = [d for d in new_docs if int(d.get("chunk_level", 0)) == 2]
        chunk_texts = [{"text": d.get("text", ""), "chunk_id": d.get("chunk_id", "")} for d in l1_chunks[:50]]
        l3_ids = [d["chunk_id"] for d in leaf_docs]

        return {
            "status": "ok",
            "filename": filename,
            "chunks": chunk_texts,
            "total_chunks": len(leaf_docs),
            "l3_ids": l3_ids,
        }

    except Exception as e:
        log.error("stage1_failed", filename=filename, error=str(e))
        return {"status": "error", "filename": filename, "error": str(e)}


def handle_graph_extract(data: dict) -> dict:
    """Stage 2: LLM 实体关系抽取。"""
    from backend.documents.graph_extractor import extract_from_l2_chunks

    filename = data["filename"]
    chunks = data.get("chunks", [])

    log.info("stage2_start", filename=filename, chunk_count=len(chunks))

    try:
        if not chunks:
            return {"status": "skip", "filename": filename, "reason": "no_chunks"}

        # extract_from_l2_chunks 是 async 函数，同步调用
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(extract_from_l2_chunks(chunks, filename))
        finally:
            loop.close()

        entities = [{"name": e.name, "type": e.type, "description": e.description} for e in result.entities]
        relations = [{"subject": r.subject, "predicate": r.predicate, "object": r.object} for r in result.relations]

        log.info("stage2_complete", filename=filename, entities=len(entities), relations=len(relations))

        return {
            "status": "ok",
            "filename": filename,
            "entities": entities,
            "relations": relations,
            "l3_ids": data.get("l3_ids", []),
        }

    except Exception as e:
        log.error("stage2_failed", filename=filename, error=str(e))
        return {"status": "error", "filename": filename, "error": str(e)}


def handle_vector_sync(data: dict) -> dict:
    """Stage 3: Neo4j 写入 + 增量聚类 + 摘要更新。"""
    from backend.storage.graph_ingestion import ingest_extraction_result

    filename = data["filename"]
    entities = data.get("entities", [])
    relations = data.get("relations", [])
    l3_ids = data.get("l3_ids", [])

    log.info("stage3_start", filename=filename)

    try:
        if not entities:
            return {"status": "skip", "filename": filename, "reason": "no_entities"}

        # 1. 写入 Neo4j
        ingest_result = ingest_extraction_result(entities, relations, l3_ids)
        log.info("stage3_neo4j_done", filename=filename, **ingest_result)

        # 2. 增量聚类
        cluster_result = {"patched": 0, "reclustered": 0, "affected_communities": []}
        try:
            from backend.graph.incremental_clustering import incremental_cluster_after_ingest
            cluster_result = incremental_cluster_after_ingest(filename)
            log.info("stage3_cluster_done", filename=filename, **cluster_result)
        except Exception as e:
            log.warning("stage3_cluster_failed", filename=filename, error=str(e))

        # 3. 定向摘要更新
        summary_result = {"updated": 0}
        try:
            from backend.pipeline.summary_updater import run_summary_update_cycle
            summary_result = run_summary_update_cycle()
            log.info("stage3_summary_done", filename=filename, updated=summary_result.get("updated", 0))
        except Exception as e:
            log.warning("stage3_summary_failed", filename=filename, error=str(e))

        return {
            "status": "ok",
            "filename": filename,
            "neo4j": ingest_result,
            "clustering": cluster_result,
            "summary": summary_result,
        }

    except Exception as e:
        log.error("stage3_failed", filename=filename, error=str(e))
        return {"status": "error", "filename": filename, "error": str(e)}


def run_stream_consumer(max_iterations: int = None):
    """主消费循环：从三个 Stream 消费消息。"""
    sq = get_stream_queue()
    if not sq or not sq._get_client():
        log.warning("stream_consumer_unavailable", reason="Redis not available")
        return

    handlers = {
        DOC_INGEST: (PARSER_GROUP, handle_doc_ingest, GRAPH_EXTRACT),
        GRAPH_EXTRACT: (EXTRACTOR_GROUP, handle_graph_extract, VECTOR_SYNC),
        VECTOR_SYNC: (SYNCER_GROUP, handle_vector_sync, None),
    }

    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        processed = 0

        for stream, (group, handler, next_stream) in handlers.items():
            messages = sq.consume(stream, group, consumer="consumer_1", count=1, block_ms=1000)
            for msg in messages:
                try:
                    data = msg["data"]
                    result = handler(data)

                    if result.get("status") == "error":
                        # 重试或死信
                        sq.try_dead_letter(stream, group, msg["id"], {"error": result.get("error", "")})
                    elif next_stream and result.get("status") == "ok":
                        # 发布到下一阶段
                        sq.ack_and_publish(stream, group, msg["id"], next_stream, result)
                    else:
                        sq.ack(stream, group, msg["id"])

                    processed += 1
                except Exception as e:
                    log.error("consumer_error", stream=stream, error=str(e))

        if processed == 0 and max_iterations is None:
            time.sleep(1)  # 无消息时短暂休眠

    log.info("consumer_stopped", iterations=iteration)
