"""文本→视觉描述语义检索通道。"""
from backend.embedding.service import EmbeddingService
from backend.milvus.client import MilvusManager

_embedding = EmbeddingService()
_milvus = MilvusManager()


def retrieve_visual(query: str, top_k: int = 5) -> list[dict]:
    """检索与查询相关的图片/表格描述。"""
    try:
        query_vec = _embedding.get_embeddings([query])[0]
        results = _milvus.dense_retrieve(
            query_vec, top_k=top_k,
            filter_expr='is_media == true',
        )
        return results if results else []
    except Exception:
        return []
