"""检索降级策略：Neo4j 超时 → 纯向量双路召回。"""
from backend.observability import get_logger

log = get_logger("ragent.ha")


def safe_graph_search(query: str, top_k: int = 5) -> dict:
    """包裹图搜索，超时时自动降级为纯向量检索。"""
    try:
        from backend.rag.graph_retriever import local_graph_search
        return local_graph_search(query, top_k)
    except Exception as e:
        log.warning("graph_search_degraded", query=query[:100], error=str(e))
        from backend.rag.utils import retrieve_documents
        result = retrieve_documents(query, top_k=top_k)
        return {
            **result,
            "mode": "degraded_dense_sparse",
            "degradation_reason": str(e)[:200],
        }
