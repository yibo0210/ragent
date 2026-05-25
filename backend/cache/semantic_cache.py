"""语义缓存层：Milvus ANN + cosine 匹配，MySQL 存储。"""
import os
import hashlib
import numpy as np

from backend.milvus.client import MilvusManager
from backend.embedding.service import EmbeddingService
from backend.storage.database import SessionLocal
from backend.storage.models import QueryCacheStore

CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

_milvus = MilvusManager()
_embedding = EmbeddingService()


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def query_cache(query: str) -> dict | None:
    """查询语义缓存。命中返回 dict，未命中返回 None。"""
    query_vector = _embedding.get_embeddings([query])[0]

    _milvus.init_cache_collection()
    candidates = _milvus.search_cache(query_vector, top_k=3)

    best_score = 0.0
    best_hash = None
    for c in candidates:
        entity = c.get("entity", {})
        emb = entity.get("embedding", [])
        if len(emb) > 0:
            score = float(np.dot(query_vector, emb) / (
                np.linalg.norm(query_vector) * np.linalg.norm(emb)
            ))
        else:
            score = c.get("distance", 0)
        if score > best_score:
            best_score = score
            best_hash = entity.get("query_hash", "")

    if best_score < CACHE_SIMILARITY_THRESHOLD:
        return None

    with SessionLocal() as session:
        entry = session.query(QueryCacheStore).filter_by(query_hash=best_hash).first()
        if entry:
            entry.hit_count += 1
            session.commit()
            return {
                "response": entry.response_text,
                "source_doc": entry.source_doc,
                "similarity": round(float(best_score), 4),
                "hit_count": entry.hit_count,
                "cached": True,
            }
    return None


def write_cache(query: str, response: str, source_doc: str = "") -> dict:
    """写入语义缓存（Milvus + MySQL）。"""
    query_vector = _embedding.get_embeddings([query])[0]
    query_hash = _hash_query(query)

    _milvus.init_cache_collection()
    insert_result = _milvus.insert_cache(query_vector, query_hash, query, source_doc)
    vector_id = str(insert_result.get("ids", [None])[0]) if insert_result else ""

    with SessionLocal() as session:
        existing = session.query(QueryCacheStore).filter_by(query_hash=query_hash).first()
        if existing:
            existing.response_text = response
            existing.hit_count = 1
        else:
            session.add(QueryCacheStore(
                query_hash=query_hash,
                vector_id=vector_id,
                response_text=response,
                source_doc=source_doc,
            ))
        session.commit()

    return {"query_hash": query_hash, "vector_id": vector_id, "status": "cached"}
