"""v12/v17 Query Profiler — lightweight intent classifier.

L3 complexity (v12) + 6 query types (v17).
Hybrid scoring: keyword 60% + Embedding cosine similarity 40%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from backend.observability import get_logger

logger = get_logger("query_profiler")

# ---------------------------------------------------------------------------
# v17: 6 query types
# ---------------------------------------------------------------------------
FACTOID = "factoid"
ENTITY_RELATION = "entity_relation"
MULTI_HOP = "multi_hop"
GLOBAL_SUMMARY = "global_summary"
TEMPORAL = "temporal"
COMPARISON = "comparison"

# ---------------------------------------------------------------------------
# L1/L2/L3 keywords (v12, kept for backward compat)
# ---------------------------------------------------------------------------
_L1_KEYWORDS: list[str] = [
    "你好", "hi", "hello", "谢谢", "天气",
]

_L2_KEYWORDS: list[str] = [
    "什么是", "是什么", "关系", "关联", "依赖", "影响", "区别", "对比", "比较",
    "为什么", "原因", "如何实现", "原理", "哪些组件", "多跳", "推理",
]

_L3_KEYWORDS: list[str] = [
    "总结", "综述", "全面", "整体", "全局", "全景", "所有", "全部",
    "主要", "架构是怎样的", "概览",
]

# ---------------------------------------------------------------------------
# v17: 6-type keywords
# ---------------------------------------------------------------------------
_TYPE_KEYWORDS: dict[str, list[str]] = {
    FACTOID: ["what is", "define", "meaning of", "definition", "explain", "describe"],
    ENTITY_RELATION: ["who", "which company", "founded", "invested", "acquired",
                      "CEO", "owner", "subsidiary", "competitor", "partner of"],
    MULTI_HOP: ["chain", "path", "via", "through", "connected", "network", "trace",
                "how many", "what is the relationship between"],
    GLOBAL_SUMMARY: ["summarize", "overview", "themes", "summary", "overall",
                     "key points", "key findings", "recap"],
    TEMPORAL: ["in 202", "before 202", "after 202", "last year", "previous",
               "q1", "q2", "q3", "q4", "quarter", "last month", "fiscal"],
    COMPARISON: ["compare", "difference", "versus", "vs", "better", "contrast",
                 "pros and cons", "advantages", "disadvantages", "which is better"],
}

# ---------------------------------------------------------------------------
# v17: 6-type prototypes (4 per type, for embedding similarity)
# ---------------------------------------------------------------------------
_TYPE_PROTOTYPES: dict[str, list[str]] = {
    FACTOID: [
        "What is Redis?",
        "What is Kafka used for?",
        "Define FastAPI framework.",
        "Explain what a database index does.",
    ],
    ENTITY_RELATION: [
        "Who founded OpenAI?",
        "Which companies are invested by Tencent?",
        "Who is the CEO of Microsoft?",
        "What are the major products of Apple?",
    ],
    MULTI_HOP: [
        "Which company acquired the startup that developed Kubernetes?",
        "Find competitors of the company that partnered with our main supplier.",
        "Which organizations collaborated with both OpenAI and Google?",
        "Trace the investment chain from SoftBank to ByteDance.",
    ],
    GLOBAL_SUMMARY: [
        "Summarize the entire system architecture.",
        "What are the major themes across all documents?",
        "Give me a comprehensive overview of the project.",
        "What are the key takeaways from the knowledge base?",
    ],
    TEMPORAL: [
        "Who was CEO in 2022?",
        "What happened in Q3 2023?",
        "Before 2020, which technology stack was used?",
        "Compare performance between 2021 and 2022.",
    ],
    COMPARISON: [
        "Compare GraphRAG and vanilla RAG.",
        "What are the differences between Redis and Memcached?",
        "Compare MySQL and PostgreSQL performance.",
        "Which is better for enterprise use: FastAPI or Flask?",
    ],
}

# ---------------------------------------------------------------------------
# v12: L1/L2/L3 prototypes (kept for backward compat)
# ---------------------------------------------------------------------------
_L1_PROTOTYPES: list[str] = [
    "你好，请问你是谁？",
    "Python 是什么？",
    "今天天气怎么样？",
    "谢谢你的帮助",
]

_L2_PROTOTYPES: list[str] = [
    "Milvus 和 Neo4j 之间有什么关系？",
    "GraphRAG 依赖哪些组件来实现多跳推理？",
    "LangChain 和 LangGraph 的区别是什么？",
    "为什么系统使用 RRF 融合多路检索？",
]

_L3_PROTOTYPES: list[str] = [
    "系统整体技术架构是怎样的？请全面总结。",
    "所有文档中的方法有什么区别？",
    "请综述当前知识库中的核心技术栈。",
    "全局概览整个系统的模块组成和数据流。",
]

# Module-level cache for prototype embeddings
_prototype_embeddings: Optional[dict[str, list[list[float]]]] = None
_type_prototype_embeddings: Optional[dict[str, list[list[float]]]] = None

_KEYWORD_WEIGHT = 0.6
_EMBEDDING_WEIGHT = 0.4

# v17: graph_hops per query type
_GRAPH_HOPS_MAP: dict[str, int] = {
    FACTOID: 0,
    ENTITY_RELATION: 1,
    MULTI_HOP: 3,
    GLOBAL_SUMMARY: 0,
    TEMPORAL: 1,
    COMPARISON: 1,
}

# v17: type → L1/L2/L3 level mapping (backward compat)
_TYPE_TO_LEVEL: dict[str, str] = {
    FACTOID: "L1_FACTUAL",
    ENTITY_RELATION: "L2_REASONING",
    MULTI_HOP: "L2_REASONING",
    GLOBAL_SUMMARY: "L3_MACRO_SUMMARY",
    TEMPORAL: "L2_REASONING",
    COMPARISON: "L2_REASONING",
}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_prototype_embeddings(embedding_service) -> dict[str, list[list[float]]]:
    try:
        all_queries = _L1_PROTOTYPES + _L2_PROTOTYPES + _L3_PROTOTYPES
        all_embs = embedding_service.get_embeddings(all_queries)
        n = len(_L1_PROTOTYPES)
        return {
            "L1_FACTUAL": all_embs[:n],
            "L2_REASONING": all_embs[n:2 * n],
            "L3_MACRO_SUMMARY": all_embs[2 * n:3 * n],
        }
    except Exception as e:
        logger.warning("原型 Embedding 生成失败，降级为纯关键词模式", error=str(e))
        return None


def _build_type_prototype_embeddings(embedding_service) -> dict[str, list[list[float]]]:
    try:
        result = {}
        for qtype in [FACTOID, ENTITY_RELATION, MULTI_HOP, GLOBAL_SUMMARY, TEMPORAL, COMPARISON]:
            queries = _TYPE_PROTOTYPES[qtype]
            embs = embedding_service.get_embeddings(queries)
            result[qtype] = embs
        return result
    except Exception as e:
        logger.warning("类型原型 Embedding 生成失败", error=str(e))
        return None


def warmup():
    global _prototype_embeddings, _type_prototype_embeddings
    try:
        from backend.embedding.service import EmbeddingService
        service = EmbeddingService()
        if _prototype_embeddings is None:
            _prototype_embeddings = _build_prototype_embeddings(service)
        if _type_prototype_embeddings is None:
            _type_prototype_embeddings = _build_type_prototype_embeddings(service)
        logger.info("query_profiler_warmup_complete")
    except Exception as e:
        logger.warning("query_profiler_warmup_failed", error=str(e))


@dataclass
class QueryIntent:
    level: str  # L1_FACTUAL / L2_REASONING / L3_MACRO_SUMMARY
    complexity_score: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    embedding_similarity: dict = field(default_factory=dict)
    reason: str = ""
    # v17 fields
    query_type: str = ""
    graph_hops: int = 1

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "complexity_score": round(self.complexity_score, 4),
            "matched_keywords": self.matched_keywords,
            "embedding_similarity": {
                k: round(v, 4) for k, v in self.embedding_similarity.items()
            },
            "reason": self.reason,
            "query_type": self.query_type,
            "graph_hops": self.graph_hops,
        }


class QueryProfiler:
    def __init__(self, use_embedding: bool = True):
        self.use_embedding = use_embedding
        self._embedding_service = None

    def _get_embedding_service(self):
        if self._embedding_service is None:
            from backend.embedding.service import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _keyword_score(self, query: str) -> dict[str, tuple[float, list[str]]]:
        query_lower = query.lower()
        results = {}
        for level, keywords in [
            ("L1_FACTUAL", _L1_KEYWORDS),
            ("L2_REASONING", _L2_KEYWORDS),
            ("L3_MACRO_SUMMARY", _L3_KEYWORDS),
        ]:
            matched = [kw for kw in keywords if kw in query_lower]
            score = min(len(matched) / len(keywords), 1.0) if keywords else 0.0
            results[level] = (score, matched)
        return results

    def _classify_query_type(self, query: str) -> tuple[str, list[str]]:
        """Classify a query into one of 6 types based on keyword matching.

        Temporal has priority boost: if temporal keywords match, prefer temporal
        over types that also match (e.g. 'Who was CEO in 2022?' → temporal, not entity_relation).
        """
        query_lower = query.lower().strip()
        type_scores: dict[str, int] = {}
        for qtype, keywords in _TYPE_KEYWORDS.items():
            matched = sum(1 for kw in keywords if kw in query_lower)
            if matched > 0:
                # Temporal gets +2 boost to win ties against entity_relation
                bonus = 2 if qtype == TEMPORAL else 0
                type_scores[qtype] = matched + bonus
        if not type_scores:
            return FACTOID, []
        best_type = max(type_scores, key=type_scores.get)
        matched_kws = [kw for kw in _TYPE_KEYWORDS[best_type] if kw in query_lower]
        return best_type, matched_kws

    def _embedding_score(self, query: str) -> dict[str, float]:
        global _prototype_embeddings
        if not self.use_embedding:
            return {"L1_FACTUAL": 0.0, "L2_REASONING": 0.0, "L3_MACRO_SUMMARY": 0.0}
        try:
            service = self._get_embedding_service()
            if _prototype_embeddings is None:
                _prototype_embeddings = _build_prototype_embeddings(service)
            if _prototype_embeddings is None:
                return {"L1_FACTUAL": 0.0, "L2_REASONING": 0.0, "L3_MACRO_SUMMARY": 0.0}
            query_emb = service.get_embeddings([query])[0]
            results = {}
            for level in ["L1_FACTUAL", "L2_REASONING", "L3_MACRO_SUMMARY"]:
                similarities = [
                    _cosine_similarity(query_emb, proto_emb)
                    for proto_emb in _prototype_embeddings[level]
                ]
                results[level] = sum(similarities) / len(similarities) if similarities else 0.0
            return results
        except Exception as e:
            logger.warning("Embedding 相似度计算失败，降级为纯关键词模式", error=str(e))
            return {"L1_FACTUAL": 0.0, "L2_REASONING": 0.0, "L3_MACRO_SUMMARY": 0.0}

    def _type_embedding_score(self, query: str) -> dict[str, float]:
        """v17: Embedding similarity against 6-type prototypes."""
        global _type_prototype_embeddings
        if not self.use_embedding:
            return {t: 0.0 for t in [FACTOID, ENTITY_RELATION, MULTI_HOP, GLOBAL_SUMMARY, TEMPORAL, COMPARISON]}
        try:
            service = self._get_embedding_service()
            if _type_prototype_embeddings is None:
                _type_prototype_embeddings = _build_type_prototype_embeddings(service)
            if _type_prototype_embeddings is None:
                return {t: 0.0 for t in [FACTOID, ENTITY_RELATION, MULTI_HOP, GLOBAL_SUMMARY, TEMPORAL, COMPARISON]}
            query_emb = service.get_embeddings([query])[0]
            results = {}
            for qtype in [FACTOID, ENTITY_RELATION, MULTI_HOP, GLOBAL_SUMMARY, TEMPORAL, COMPARISON]:
                prototypes = _type_prototype_embeddings.get(qtype, [])
                if not prototypes:
                    results[qtype] = 0.0
                else:
                    similarities = [_cosine_similarity(query_emb, pe) for pe in prototypes]
                    results[qtype] = sum(similarities) / len(similarities)
            return results
        except Exception:
            return {t: 0.0 for t in [FACTOID, ENTITY_RELATION, MULTI_HOP, GLOBAL_SUMMARY, TEMPORAL, COMPARISON]}

    def profile(self, query: str) -> QueryIntent:
        if not query or not query.strip():
            return QueryIntent(
                level="L1_FACTUAL", query_type=FACTOID, complexity_score=0.0,
                matched_keywords=[], embedding_similarity={},
                reason="查询为空，默认归类为简单事实", graph_hops=0,
            )

        # L1/L2/L3 keyword scoring
        kw_scores = self._keyword_score(query)
        emb_scores = self._embedding_score(query)

        final_scores = {}
        for level in ["L1_FACTUAL", "L2_REASONING", "L3_MACRO_SUMMARY"]:
            kw_s, matched = kw_scores[level]
            emb_s = emb_scores[level]
            final_scores[level] = (_KEYWORD_WEIGHT * kw_s + _EMBEDDING_WEIGHT * emb_s, matched)

        if len(query.strip()) < 5:
            best_level = "L1_FACTUAL"
        else:
            best_level = max(final_scores, key=lambda k: final_scores[k][0])
        best_score, best_keywords = final_scores[best_level]

        # v17: 6-type classification
        best_type, type_keywords = self._classify_query_type(query)
        type_emb_scores = self._type_embedding_score(query)

        # Refine type using embedding
        if any(v > 0.4 for v in type_emb_scores.values()):
            best_emb_type = max(type_emb_scores, key=type_emb_scores.get)
            if type_emb_scores[best_emb_type] > 0.5:
                best_type = best_emb_type
                type_keywords = _TYPE_KEYWORDS.get(best_type, [])

        graph_hops = _GRAPH_HOPS_MAP.get(best_type, 1)

        # Complexity score
        level_complexity = {"L1_FACTUAL": 0.2, "L2_REASONING": 0.6, "L3_MACRO_SUMMARY": 1.0}
        complexity = min(level_complexity[best_level] * (0.5 + best_score), 1.0)

        parts = []
        if best_keywords:
            parts.append(f"L-keywords: {', '.join(best_keywords[:3])}")
        if type_keywords:
            parts.append(f"Type={best_type}")
        reason = "; ".join(parts) if parts else "未匹配到明确特征"

        return QueryIntent(
            level=best_level,
            complexity_score=round(complexity, 4),
            matched_keywords=best_keywords,
            embedding_similarity=emb_scores,
            reason=reason,
            query_type=best_type,
            graph_hops=graph_hops,
        )
