"""文档生命周期管理：软删除、chunk ID 查询。"""
from datetime import datetime, timezone
from sqlalchemy import update, select
from backend.storage.database import SessionLocal
from backend.storage.models import ParentChunk, DocumentIndex


def list_active_documents(tenant_id: int) -> list[dict]:
    """列出指定租户下所有活跃文档。"""
    with SessionLocal() as session:
        docs = session.query(DocumentIndex).filter(
            DocumentIndex.is_deleted == False,
            DocumentIndex.tenant_id == tenant_id,
        ).order_by(DocumentIndex.updated_at.desc()).all()
        return [
            {
                "filename": d.filename,
                "file_type": d.filename.split(".")[-1].upper() if "." in d.filename else "Unknown",
                "chunk_count": d.chunk_count,
                "uploaded_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]


def get_chunk_ids_by_filename(filename: str, include_deleted: bool = False) -> list[str]:
    """获取文档的所有 L3 chunk ID。"""
    with SessionLocal() as session:
        stmt = select(ParentChunk.chunk_id).where(
            ParentChunk.filename == filename,
            ParentChunk.chunk_level == 3,
        )
        if not include_deleted:
            stmt = stmt.where(ParentChunk.is_deleted == False)
        rows = session.execute(stmt).scalars().all()
        return [r[0] if isinstance(r, tuple) else r for r in rows]


def mark_document_deleted(filename: str, tenant_id: int = None) -> dict:
    """软删除文档：标记 ParentChunk + DocumentIndex（按租户隔离）。"""
    with SessionLocal() as session:
        now = datetime.now(timezone.utc)

        # 标记 ParentChunk
        stmt = (
            update(ParentChunk)
            .where(ParentChunk.filename == filename, ParentChunk.is_deleted == False)
            .values(is_deleted=True, version=ParentChunk.version + 1, updated_at=now)
        )
        result = session.execute(stmt)

        # 标记 DocumentIndex（按租户隔离）
        query = session.query(DocumentIndex).filter_by(filename=filename)
        if tenant_id is not None:
            query = query.filter(DocumentIndex.tenant_id == tenant_id)
        doc = query.first()
        if doc:
            doc.is_deleted = True
            doc.version += 1
            doc.updated_at = now

        session.commit()

        # v6.0: 缓存失效
        cache_invalidated = {}
        try:
            from backend.cache.invalidation import invalidate_by_filename
            cache_invalidated = invalidate_by_filename(filename)
        except Exception as e:
            from backend.observability import get_logger
            get_logger("ragent.doc_lifecycle").warning("cache_invalidation_failed", filename=filename, error=str(e))

        return {
            "filename": filename,
            "affected_chunks": result.rowcount,
            "status": "soft_deleted",
            "deleted_at": now.isoformat(),
            "cache_invalidated": cache_invalidated,
        }


def upsert_document_index(filename: str, file_hash: str, chunk_count: int, tenant_id: int = 0) -> dict:
    """Upsert DocumentIndex: 创建/更新/跳过文档索引记录。

    Returns:
        {"action": "created"|"skipped"|"updated", "old_hash": str, "new_hash": str}
    """
    with SessionLocal() as session:
        now = datetime.now(timezone.utc)
        doc = session.query(DocumentIndex).filter_by(filename=filename).first()

        if doc is None:
            # 新文档 — INSERT
            doc = DocumentIndex(
                filename=filename,
                file_hash=file_hash,
                chunk_count=chunk_count,
                is_deleted=False,
                version=1,
                tenant_id=tenant_id,
                created_at=now,
                updated_at=now,
            )
            session.add(doc)
            session.commit()
            return {"action": "created", "old_hash": "", "new_hash": file_hash}

        if doc.file_hash == file_hash and doc.chunk_count == chunk_count:
            # 内容未变且 chunk 数量相同 — 跳过
            return {"action": "skipped", "old_hash": doc.file_hash, "new_hash": file_hash}

        if doc.file_hash == file_hash:
            # 内容未变但 chunk 数量有更新（异步管线场景：先写0再更新实际值）
            doc.chunk_count = chunk_count
            doc.updated_at = now
            session.commit()
            return {"action": "updated_chunks", "old_hash": doc.file_hash, "new_hash": file_hash}

        # 内容变化 — UPDATE
        old_hash = doc.file_hash
        doc.file_hash = file_hash
        doc.chunk_count = chunk_count
        doc.is_deleted = False
        doc.version += 1
        doc.tenant_id = tenant_id
        doc.updated_at = now
        session.commit()
        return {"action": "updated", "old_hash": old_hash, "new_hash": file_hash}
