"""API 接口层模块

提供会话管理、智能问答（含流式响应）、文档上传/管理/删除等接口。
所有接口通过 FastAPI 的 APIRouter 注册。
"""
import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.agent.brain import chat_with_agent, chat_with_agent_stream, resume_hitl_graph, storage
from backend.storage.cache import cache
from backend.documents.loader import DocumentLoader
from backend.embedding.service import EmbeddingService
from backend.milvus.client import MilvusManager
from backend.milvus.writer import MilvusWriter
from backend.storage.parent_chunk_store import ParentChunkStore
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    HitlResumeRequest,
    MessageInfo,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)

# __file__ = backend/api/routes.py
# .parent.parent.parent = project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "documents"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_manager = MilvusManager()
embedding_service = EmbeddingService()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

router = APIRouter()


# ====================== 会话管理接口 ======================
#获取会话历史消息 从 storage 读取指定会话的消息，封装为 MessageInfo 列表
@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str):
    try:
        messages = [
            MessageInfo(
                type=msg["type"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                rag_trace=msg.get("rag_trace"),
                agent_trace=msg.get("agent_trace"),
            )
            for msg in storage.get_session_messages(session_id)
        ]
        return SessionMessagesResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#列出所有会话 按更新时间倒序返回会话列表
@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    try:
        sessions = [SessionInfo(**item) for item in storage.list_session_infos()]
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#删除指定会话 调用 storage 删除会话，不存在则抛 404
@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str):
    try:
        deleted = storage.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#普通问答 接收 ChatRequest（消息 + 会话 ID），调用 chat_with_agent 返回完整响应
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id or "default_session"
        if cache.is_locked(session_id):
            raise HTTPException(status_code=423, detail="会话处于人工审核等待中，请先完成审核操作")
        resp = chat_with_agent(request.message, session_id)
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#流式问答 以 SSE（服务器发送事件）返回流式响应，设置禁用缓存 / 长连接头，异常时返回 error 类型数据
@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    session_id = request.session_id or "default_session"
    if cache.is_locked(session_id):
        raise HTTPException(status_code=423, detail="会话处于人工审核等待中，请先完成审核操作")
    async def event_generator():
        try:
            async for chunk in chat_with_agent_stream(request.message, session_id):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

#列出所有已上传文档 查询 Milvus 中所有文档，按文件名聚合统计 chunk 数量，返回 DocumentInfo 列表
@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    try:
        milvus_manager.init_collection()
        results = milvus_manager.query(output_fields=["filename", "file_type"], limit=10000)
        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            if filename not in file_stats:
                file_stats[filename] = {"filename": filename, "file_type": file_type, "chunk_count": 0}
            file_stats[filename]["chunk_count"] += 1
        documents = [DocumentInfo(**stats) for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

#上传并解析文档（SSE 流式返回进度）
@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="文件名为空，请重新选择文件后上传")
    file_lower = filename.lower()
    supported = (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".md", ".markdown", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
    if not file_lower.endswith(supported):
        raise HTTPException(status_code=400, detail=f"不支持的文件格式 (.{filename.rsplit('.', 1)[-1] if '.' in filename else '未知'})，仅支持 PDF、Word、Excel、Markdown、图片")

    async def event_generator():
        import asyncio
        loop = asyncio.get_event_loop()
        progress_queue = asyncio.Queue()

        def progress_callback(current, total, status):
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                {"type": "progress", "current": current, "total": total, "status": status}
            )

        try:
            # 1. 保存文件
            yield f'data: {json.dumps({"type": "progress", "stage": "saving", "current": 0, "total": 0, "status": "正在保存文件..."})}\n\n'
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            milvus_manager.init_collection()
            milvus_manager.delete(f'filename == "{filename}"')
            parent_chunk_store.delete_by_filename(filename)

            file_path = UPLOAD_DIR / filename
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            # 2. 解析文档
            yield f'data: {json.dumps({"type": "progress", "stage": "parsing", "current": 0, "total": 0, "status": "正在解析文档..."})}\n\n'
            new_docs = await loop.run_in_executor(None, loader.load_document, str(file_path), filename)
            parent_docs = [d for d in new_docs if int(d.get("chunk_level", 0)) in (1, 2)]
            leaf_docs = [d for d in new_docs if int(d.get("chunk_level", 0)) == 3]
            total_chunks = len(leaf_docs)

            yield f'data: {json.dumps({"type": "progress", "stage": "parsed", "current": 0, "total": total_chunks, "status": f"文档解析完成，共 {total_chunks} 个片段"})}\n\n'

            # 3. 存储父块
            await loop.run_in_executor(None, parent_chunk_store.upsert_documents, parent_docs)

            # 4. 向量化并写入 Milvus（后台线程 + 进度回调）
            import threading
            done_event = threading.Event()
            error_holder = [None]

            def _write():
                try:
                    milvus_writer.write_documents(leaf_docs, progress_callback=progress_callback)
                except Exception as e:
                    error_holder[0] = e
                finally:
                    done_event.set()

            thread = threading.Thread(target=_write)
            thread.start()

            # 持续读取进度并 yield
            while not done_event.is_set():
                try:
                    progress = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                    yield f'data: {json.dumps(progress)}\n\n'
                except asyncio.TimeoutError:
                    pass

            # 排空剩余进度
            while not progress_queue.empty():
                progress = progress_queue.get_nowait()
                yield f'data: {json.dumps(progress)}\n\n'

            if error_holder[0]:
                raise error_holder[0]

            # 5. 图谱抽取（使用 L1 块，语义更完整，利于关系提取）
            l1_chunks = [d for d in new_docs if int(d.get("chunk_level", 0)) == 1]
            l3_ids = [d["chunk_id"] for d in leaf_docs]
            if not l1_chunks:
                l1_chunks = [d for d in new_docs if int(d.get("chunk_level", 0)) == 2]
            if l1_chunks:
                try:
                    from backend.documents.graph_extractor import extract_from_l2_chunks
                    from backend.storage.graph_ingestion import ingest_extraction_result

                    yield f'data: {json.dumps({"type": "progress", "stage": "graph", "current": 0, "total": len(l1_chunks), "status": "正在提取知识图谱..."})}\n\n'
                    result = await extract_from_l2_chunks(l1_chunks, filename)
                    stats = await loop.run_in_executor(
                        None, ingest_extraction_result,
                        result.entities, result.relations, l3_ids,
                    )
                    graph_status = f"知识图谱: {stats['entities']} 实体, {stats['relations']} 关系"
                    yield f'data: {json.dumps({"type": "progress", "stage": "graph", "current": len(l1_chunks), "total": len(l1_chunks), "status": graph_status})}\n\n'
                except Exception as e:
                    print(f"[GRAPH] Extraction failed (non-fatal): {e}")
                    yield f'data: {json.dumps({"type": "progress", "stage": "graph", "status": f"图谱抽取跳过: {e}"})}\n\n'

            # 6. 完成
            yield f'data: {json.dumps({"type": "complete", "filename": filename, "chunks": total_chunks, "message": f"成功上传：{filename}"})}\n\n'

        except Exception as e:
            err_detail = str(e).encode("utf-8", errors="replace").decode("utf-8")
            yield f'data: {json.dumps({"type": "error", "message": f"上传失败: {err_detail}"})}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

#删除指定文档 — 跨库级联软删除（MySQL + Milvus + Neo4j）
@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str):
    from backend.storage.doc_lifecycle import mark_document_deleted, get_chunk_ids_by_filename
    from backend.storage.graph_cleanup import full_cascade_cleanup

    # 1. 数据库软删除
    result = mark_document_deleted(filename)
    if result["affected_chunks"] == 0:
        raise HTTPException(status_code=404, detail=f"文档 '{filename}' 不存在或已删除")

    # 2. 获取 chunk IDs 用于 Milvus + Neo4j 清理
    chunk_ids = get_chunk_ids_by_filename(filename, include_deleted=True)

    # 3. Milvus 向量删除
    milvus_manager.init_collection()
    milvus_deleted = milvus_manager.delete_by_chunk_ids(chunk_ids)

    # 4. Neo4j 图清理
    graph_result = full_cascade_cleanup(chunk_ids)

    return DocumentDeleteResponse(
        filename=filename,
        status=result["status"],
        affected_chunks=result["affected_chunks"],
        milvus_deleted=milvus_deleted,
        graph_edges_updated=graph_result["edges_updated"],
        graph_empty_edges_deleted=graph_result["empty_edges_deleted"],
        graph_orphan_nodes_deleted=graph_result["orphan_nodes_deleted"],
    )


# ====================== HITL 中断恢复接口 ======================
@router.post("/chat/hitl/resume")
async def hitl_resume_endpoint(request: HitlResumeRequest):
    """恢复因 HITL 中断而挂起的图执行，流式返回恢复后的回答。"""
    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        resume_hitl_graph(
            session_id=request.session_id,
            action=request.action,
            modified_input=request.modified_input or "",
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
