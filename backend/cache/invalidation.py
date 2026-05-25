"""文档删除 → 语义缓存失效。"""
from backend.milvus.client import MilvusManager
from backend.storage.database import SessionLocal
from backend.storage.models import QueryCacheStore


def invalidate_by_filename(filename: str) -> dict:
    """级联清除与文档相关的所有语义缓存。"""
    milvus = MilvusManager()
    milvus.init_cache_collection()
    milvus_deleted = milvus.delete_cache_by_source(filename)

    with SessionLocal() as session:
        entries = session.query(QueryCacheStore).filter_by(source_doc=filename).all()
        mysql_deleted = len(entries)
        for e in entries:
            session.delete(e)
        session.commit()

    return {"milvus_deleted": milvus_deleted, "mysql_deleted": mysql_deleted}
