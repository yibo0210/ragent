"""v12 Query Profiler — 轻量级意图分类器。

在 Supervisor LLM 之前插入的规则+Embedding 混合分类器，
将查询按复杂度分为三级，减少不必要的 LLM 调用。

分级策略：
- L1_FACTUAL: 简单事实/闲聊 → direct_answer
- L2_REASONING: 多跳逻辑推理 → local_graph_search + rag_specialist
- L3_MACRO_SUMMARY: 宏观全局总结 → global_graph_search

综合打分：关键词 60% + Embedding 余弦相似度 40%
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from backend.observability import get_logger

logger = get_logger("query_profiler")

# ---------------------------------------------------------------------------
# 关键词定义
# ---------------------------------------------------------------------------
_L1_KEYWORDS: list[str] = [
    "你好", "hi", "hello", "谢谢", "是什么", "什么是", "天气",
]

_L2_KEYWORDS: list[str] = [
    "关系", "关联", "依赖", "影响", "区别", "对比", "比较",
    "为什么", "原因", "如何实现", "原理", "哪些组件", "多跳", "推理",
]

_L3_KEYWORDS: list[str] = [
    "总结", "综述", "全面", "整体", "全局", "全景", "所有", "全部",
    "主要", "架构是怎样的", "概览",
]

# ---------------------------------------------------------------------------
# 原型查询（每级 4 条，用于 Embedding 余弦相似度计算）
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

# 模块级缓存：原型 Embedding（延迟初始化）
_prototype_embeddings: Optional[dict[str, list[list[float]]]] = None

# 关键词权重
_KEYWORD_WEIGHT = 0.6
_EMBEDDING_WEIGHT = 0.4


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_prototype_embeddings(embedding_service) -> dict[str, list[list[float]]]:
    """为各级别原型查询生成 Embedding 并缓存（单次 API 调用）。"""
    try:
        all_queries = _L1_PROTOTYPES + _L2_PROTOTYPES + _L3_PROTOTYPES
        all_embs = embedding_service.get_embeddings(all_queries)
        n = len(_L1_PROTOTYPES)
        return {
            "L1_FACTUAL": all_embs[:n],
            "L2_REASONING": all_embs[n:2*n],
            "L3_MACRO_SUMMARY": all_embs[2*n:3*n],
        }
    except Exception as e:
        logger.warning("原型 Embedding 生成失败，降级为纯关键词模式", error=str(e))
        return None


def warmup():
    """预热原型 Embedding 缓存（应在应用启动时调用）。"""
    global _prototype_embeddings
    if _prototype_embeddings is not None:
        return
    try:
        from backend.embedding.service import EmbeddingService
        service = EmbeddingService()
        _prototype_embeddings = _build_prototype_embeddings(service)
        logger.info("query_profiler_warmup_complete")
    except Exception as e:
        logger.warning("query_profiler_warmup_failed", error=str(e))


@dataclass
class QueryIntent:
    """查询意图分类结果。"""

    level: str  # L1_FACTUAL / L2_REASONING / L3_MACRO_SUMMARY
    complexity_score: float = 0.0  # 0.0~1.0
    matched_keywords: list[str] = field(default_factory=list)
    embedding_similarity: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        """序列化为字典，方便日志和上游消费。"""
        return {
            "level": self.level,
            "complexity_score": round(self.complexity_score, 4),
            "matched_keywords": self.matched_keywords,
            "embedding_similarity": {
                k: round(v, 4) for k, v in self.embedding_similarity.items()
            },
            "reason": self.reason,
        }


class QueryProfiler:
    """轻量级查询意图分类器。

    使用规则关键词 + Embedding 余弦相似度将查询分为三级，
    综合打分：关键词 60% + Embedding 40%。
    """

    def __init__(self, use_embedding: bool = True):
        """初始化分类器。

        Args:
            use_embedding: 是否启用 Embedding 相似度计算。
                设为 False 时为纯关键词模式（适用于测试或 Embedding 服务不可用场景）。
        """
        self.use_embedding = use_embedding
        self._embedding_service = None

    def _get_embedding_service(self):
        """延迟初始化 EmbeddingService（避免循环导入和不必要的初始化）。"""
        if self._embedding_service is None:
            from backend.embedding.service import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _keyword_score(self, query: str) -> dict[str, tuple[float, list[str]]]:
        """关键词匹配打分。

        Returns:
            各级别的得分和匹配到的关键词列表:
            {"L1_FACTUAL": (score, [kw, ...]), "L2_REASONING": (...), "L3_MACRO_SUMMARY": (...)}
        """
        query_lower = query.lower()
        results = {}

        for level, keywords in [
            ("L1_FACTUAL", _L1_KEYWORDS),
            ("L2_REASONING", _L2_KEYWORDS),
            ("L3_MACRO_SUMMARY", _L3_KEYWORDS),
        ]:
            matched = [kw for kw in keywords if kw in query_lower]
            # 得分 = 匹配关键词数 / 该级别关键词总数，上限 1.0
            score = min(len(matched) / len(keywords), 1.0) if keywords else 0.0
            results[level] = (score, matched)

        return results

    def _embedding_score(self, query: str) -> dict[str, float]:
        """Embedding 余弦相似度打分。

        Returns:
            各级别与用户查询的平均余弦相似度。
        """
        global _prototype_embeddings

        if not self.use_embedding:
            return {"L1_FACTUAL": 0.0, "L2_REASONING": 0.0, "L3_MACRO_SUMMARY": 0.0}

        try:
            service = self._get_embedding_service()

            # 延迟初始化原型 Embedding 缓存
            if _prototype_embeddings is None:
                _prototype_embeddings = _build_prototype_embeddings(service)

            if _prototype_embeddings is None:
                # Embedding 生成失败，降级
                return {"L1_FACTUAL": 0.0, "L2_REASONING": 0.0, "L3_MACRO_SUMMARY": 0.0}

            # 计算用户查询的 Embedding
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

    def profile(self, query: str) -> QueryIntent:
        """对用户查询进行意图分类。

        Args:
            query: 用户输入的查询文本。

        Returns:
            QueryIntent 分类结果。
        """
        # 空查询强制 L1
        if not query or not query.strip():
            return QueryIntent(
                level="L1_FACTUAL",
                complexity_score=0.0,
                matched_keywords=[],
                embedding_similarity={},
                reason="查询为空，默认归类为简单事实",
            )

        # 关键词打分
        kw_scores = self._keyword_score(query)

        # Embedding 打分
        emb_scores = self._embedding_score(query)

        # 综合打分：关键词 60% + Embedding 40%
        final_scores = {}
        for level in ["L1_FACTUAL", "L2_REASONING", "L3_MACRO_SUMMARY"]:
            kw_s, matched = kw_scores[level]
            emb_s = emb_scores[level]
            final_scores[level] = (
                _KEYWORD_WEIGHT * kw_s + _EMBEDDING_WEIGHT * emb_s,
                matched,
            )

        # 选择得分最高的级别；短查询（< 5 字符）强制 L1
        if len(query.strip()) < 5:
            best_level = "L1_FACTUAL"
        else:
            best_level = max(final_scores, key=lambda k: final_scores[k][0])
        best_score, best_keywords = final_scores[best_level]

        # 复杂度分数归一化到 0.0~1.0（L1=低，L2=中，L3=高）
        level_complexity = {"L1_FACTUAL": 0.2, "L2_REASONING": 0.6, "L3_MACRO_SUMMARY": 1.0}
        # 结合匹配强度调整
        base = level_complexity[best_level]
        complexity = min(base * (0.5 + best_score), 1.0)

        # 构建原因说明
        parts = []
        if best_keywords:
            parts.append(f"关键词匹配: {', '.join(best_keywords)}")
        if any(v > 0 for v in emb_scores.values()):
            top_emb_level = max(emb_scores, key=emb_scores.get)
            parts.append(f"Embedding 最高相似度: {top_emb_level}={emb_scores[top_emb_level]:.3f}")
        reason = "; ".join(parts) if parts else "未匹配到明确特征"

        return QueryIntent(
            level=best_level,
            complexity_score=round(complexity, 4),
            matched_keywords=best_keywords,
            embedding_similarity=emb_scores,
            reason=reason,
        )
