"""arq worker: async document ingestion pipeline.

Runs as a standalone process via ``arq backend.pipeline.ingestion_worker.WorkerSettings``.
Handles its own DB / service initialization (not dependent on FastAPI startup).
"""
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path so ``backend.*`` imports resolve.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Lazy singletons — initialised once in on_startup
# ---------------------------------------------------------------------------
_loader = None
_parent_chunk_store = None
_milvus_writer = None


def _get_loader():
    global _loader
    if _loader is None:
        from backend.documents.loader import DocumentLoader
        _loader = DocumentLoader()
    return _loader


def _get_parent_chunk_store():
    global _parent_chunk_store
    if _parent_chunk_store is None:
        from backend.storage.parent_chunk_store import ParentChunkStore
        _parent_chunk_store = ParentChunkStore()
    return _parent_chunk_store


def _get_milvus_writer():
    global _milvus_writer
    if _milvus_writer is None:
        from backend.embedding.service import EmbeddingService
        from backend.milvus.client import MilvusManager
        from backend.milvus.writer import MilvusWriter
        embedding_service = EmbeddingService()
        milvus_manager = MilvusManager()
        _milvus_writer = MilvusWriter(
            embedding_service=embedding_service,
            milvus_manager=milvus_manager,
        )
    return _milvus_writer


# ---------------------------------------------------------------------------
# Core task
# ---------------------------------------------------------------------------

async def run_ingestion_task(
    ctx, filename: str, file_path: str, file_hash: str,
    tenant_id: int = 0, access_level: int = 1
):
    """Replicate the full ingestion pipeline that currently lives in routes.py.

    Steps:
    1. Clean up old data (Milvus, parent chunks, graph)
    2. Parse and chunk via DocumentLoader
    3. Store parent chunks via ParentChunkStore
    4. Vectorize and write to Milvus via MilvusWriter
    5. Graph extraction from L1 chunks
    6. Ingest into Neo4j
    7. Update DocumentIndex with final chunk count
    """
    from backend.milvus.client import MilvusManager
    from backend.storage.graph_cleanup import cleanup_by_filename
    from backend.storage.graph_ingestion import ingest_extraction_result
    from backend.storage.doc_lifecycle import upsert_document_index

    loop = asyncio.get_event_loop()

    # 1. Clean up old data
    milvus_mgr = MilvusManager()
    milvus_mgr.init_collection()
    milvus_mgr.delete(f'filename == "{filename}"')

    parent_store = _get_parent_chunk_store()
    parent_store.delete_by_filename(filename)

    cleanup_by_filename(filename)

    # 2. Parse and chunk
    loader = _get_loader()
    new_docs = await loop.run_in_executor(None, loader.load_document, file_path, filename)
    parent_docs = [d for d in new_docs if int(d.get("chunk_level", 0)) in (1, 2)]
    leaf_docs = [d for d in new_docs if int(d.get("chunk_level", 0)) == 3]
    total_chunks = len(leaf_docs)

    # 2.1 Inject tenant_id and access_level into doc dicts
    for doc in parent_docs:
        doc["tenant_id"] = tenant_id
    for doc in leaf_docs:
        doc["tenant_id"] = tenant_id
        doc["access_level"] = access_level

    # 3. Store parent chunks
    await loop.run_in_executor(None, parent_store.upsert_documents, parent_docs)

    # 4. Vectorize and write to Milvus
    writer = _get_milvus_writer()
    await loop.run_in_executor(None, writer.write_documents, leaf_docs)

    # 5. Graph extraction (use L1 chunks for richer semantics)
    l1_chunks = [d for d in new_docs if int(d.get("chunk_level", 0)) == 1]
    l3_ids = [d["chunk_id"] for d in leaf_docs]
    if not l1_chunks:
        l1_chunks = [d for d in new_docs if int(d.get("chunk_level", 0)) == 2]
    if l1_chunks:
        try:
            from backend.documents.graph_extractor import extract_from_l2_chunks
            result = await extract_from_l2_chunks(l1_chunks, filename)
            await loop.run_in_executor(
                None, ingest_extraction_result,
                result.entities, result.relations, l3_ids,
                tenant_id,
            )
        except Exception as e:
            print(f"[WORKER] Graph extraction failed (non-fatal): {e}")

        # v13: 增量图聚类
        try:
            from backend.graph.incremental_clustering import incremental_cluster_after_ingest
            cluster_result = incremental_cluster_after_ingest(filename)
            print(f"[v13] 增量聚类完成: patched={cluster_result['patched']}, reclustered={cluster_result['reclustered']}")
        except Exception as e:
            print(f"[v13] 增量聚类失败（非致命）: {e}")

        # v13: 定向摘要更新
        try:
            from backend.pipeline.summary_updater import run_summary_update_cycle
            summary_result = run_summary_update_cycle()
            if summary_result.get("updated", 0) > 0:
                print(f"[v13] 摘要更新: {summary_result['updated']} 个社区")
        except Exception as e:
            print(f"[v13] 摘要更新失败（非致命）: {e}")

    # 6. Update document index with final chunk count
    upsert_document_index(filename, file_hash, total_chunks, tenant_id=tenant_id)

    return {"filename": filename, "chunks": total_chunks}


# ---------------------------------------------------------------------------
# arq lifecycle hooks
# ---------------------------------------------------------------------------

async def startup(ctx):
    """Initialise DB tables and Neo4j schema once when the worker starts."""
    from backend.storage.database import init_db
    from backend.storage.graph_schema import init_graph_schema

    init_db()
    init_graph_schema()
    print("[WORKER] Startup complete — DB and graph schema initialised")


async def shutdown(ctx):
    print("[WORKER] Shutting down")


# ---------------------------------------------------------------------------
# arq WorkerSettings
# ---------------------------------------------------------------------------

class WorkerSettings:
    functions = [run_ingestion_task]
    on_startup = startup
    on_shutdown = shutdown
    from backend.pipeline.task_queue import get_redis_settings
    redis_settings = get_redis_settings()
