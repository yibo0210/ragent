"""文档生命周期管理：软删除、chunk ID 查询。"""
from datetime import datetime, timezone
from sqlalchemy import update, select
from backend.storage.database import SessionLocal
from backend.storage.models import ParentChunk, DocumentIndex


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


def mark_document_deleted(filename: str) -> dict:
    """软删除文档：标记 ParentChunk + DocumentIndex。"""
    with SessionLocal() as session:
        now = datetime.now(timezone.utc)

        # 标记 ParentChunk
        stmt = (
            update(ParentChunk)
            .where(ParentChunk.filename == filename, ParentChunk.is_deleted == False)
            .values(is_deleted=True, version=ParentChunk.version + 1, updated_at=now)
        )
        result = session.execute(stmt)

        # 标记 DocumentIndex
        doc = session.query(DocumentIndex).filter_by(filename=filename).first()
        if doc:
            doc.is_deleted = True
            doc.version += 1
            doc.updated_at = now

        session.commit()

        return {
            "filename": filename,
            "affected_chunks": result.rowcount,
            "status": "soft_deleted",
            "deleted_at": now.isoformat(),
        }
