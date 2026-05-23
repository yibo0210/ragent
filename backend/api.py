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

from agent import chat_with_agent, chat_with_agent_stream, storage
from document_loader import DocumentLoader
from embedding import EmbeddingService
from milvus_client import MilvusManager
from milvus_writer import MilvusWriter
from parent_chunk_store import ParentChunkStore
from schemas import (
    ChatRequest,
    ChatResponse,
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    MessageInfo,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
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
        resp = chat_with_agent(request.message, session_id)
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#流式问答 以 SSE（服务器发送事件）返回流式响应，设置禁用缓存 / 长连接头，异常时返回 error 类型数据
@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
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

#上传并解析文档 1. 校验文件类型（仅支持 PDF/Word/Excel） 清理该文件已存在的向量 /chunk保存文件到本地加载并切分文档为不同层级 chunk 父 chunk 存入 ParentChunkStore，叶子 chunk 写入 Milvus
@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        filename = file.filename or ""
        file_lower = filename.lower()
        if not (file_lower.endswith((".pdf", ".docx", ".doc", ".xlsx", ".xls"))):
            raise HTTPException(status_code=400, detail="仅支持 PDF、Word、Excel")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        milvus_manager.init_collection()
        milvus_manager.delete(f'filename == "{filename}"')
        parent_chunk_store.delete_by_filename(filename)

        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        new_docs = loader.load_document(str(file_path), filename)
        parent_docs = [d for d in new_docs if int(d.get("chunk_level", 0)) in (1, 2)]
        leaf_docs = [d for d in new_docs if int(d.get("chunk_level", 0)) == 3]

        parent_chunk_store.upsert_documents(parent_docs)
        milvus_writer.write_documents(leaf_docs)

        return DocumentUploadResponse(
            filename=filename, chunks_processed=len(leaf_docs),
            message=f"成功上传：{filename}"
        )
    except Exception as e:
        import traceback
        err_detail = str(e).encode("utf-8", errors="replace").decode("utf-8")
        raise HTTPException(status_code=500, detail=f"上传失败: {err_detail}")

#删除指定文档 1. 从 Milvus 删除该文件的所有向量数据  从 ParentChunkStore 删除对应父 chunk
@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str):
    try:
        milvus_manager.init_collection()
        milvus_manager.delete(f'filename == "{filename}"')
        parent_chunk_store.delete_by_filename(filename)
        return DocumentDeleteResponse(filename=filename, chunks_deleted=0, message="删除成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")