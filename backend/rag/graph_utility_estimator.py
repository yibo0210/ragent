"""GraphUtilityEstimator: predict whether graph retrieval will improve answer quality.

Uses lightweight heuristics (no LLM call) to estimate P(Graph Helpful | Query).
If score < threshold, skip Neo4j entirely.
"""

from __future__ import annotations

import re


class GraphUtilityScore:
    __slots__ = ("score", "graph_hops", "skip_reason")

    def __init__(self, score: float, graph_hops: int = 1, skip_reason: str = ""):
        self.score = score
        self.graph_hops = graph_hops
        self.skip_reason = skip_reason


class GraphUtilityEstimator:
    """Estimate graph retrieval utility from query features.

    Features (weighted):
    - Entity density: count of named entities / query length (30%)
    - Relation keywords: contains "founded", "acquired", etc. (30%)
    - Reasoning keywords: "trace", "chain", "via", "through" (25%)
    - Temporal keywords: year/quarter references (15%)
    """

    ENTITY_PATTERNS = [
        r"\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|Co|Company)\b",
        r"\b(?:OpenAI|Google|Microsoft|Apple|Amazon|Meta|Tesla|Netflix)\b",
        r"\b(?:Redis|Kafka|PostgreSQL|MySQL|MongoDB|Docker|Kubernetes)\b",
    ]

    RELATION_KEYWORDS = [
        "founded", "acquired", "invested", "partnered", "merged",
        "ceo", "cto", "founder", "owner", "subsidiary", "competitor",
        "supplier", "customer", "parent company",
    ]

    REASONING_KEYWORDS = [
        "trace", "chain", "path", "via", "through", "connected",
        "relationship", "network", "graph", "linked",
    ]

    TEMPORAL_KEYWORDS = [
        "in 202", "in 2020", "in 2021", "in 2022", "in 2023",
        "before", "after", "during", "q1", "q2", "q3", "q4",
    ]

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def estimate(self, query: str, query_type: str = "") -> GraphUtilityScore:
        query_lower = query.lower()
        words = query_lower.split()
        qlen = max(len(words), 1)

        # Entity count: regex patterns + capitalized words as heuristic
        entity_count = sum(1 for p in self.ENTITY_PATTERNS if re.search(p, query))
        # Also count capitalized Proper Nouns (words that start with uppercase and are not sentence-start)
        cap_words = re.findall(r"\b[A-Z][a-z]{2,}\b", query)
        entity_count += len(cap_words)
        entity_density = min(entity_count / max(qlen, 1) * 3, 1.0)

        rel_score = min(
            sum(1 for kw in self.RELATION_KEYWORDS if kw in query_lower) / 3, 1.0
        )
        reason_score = min(
            sum(1 for kw in self.REASONING_KEYWORDS if kw in query_lower) / 3, 1.0
        )
        time_score = min(
            sum(1 for kw in self.TEMPORAL_KEYWORDS if kw in query_lower) / 2, 1.0
        )

        score = (
            entity_density * 0.30
            + rel_score * 0.30
            + reason_score * 0.25
            + time_score * 0.15
        )

        if score >= 0.55:
            graph_hops = 3
        elif score >= 0.25:
            graph_hops = 1
        else:
            graph_hops = 0

        skip_reason = ""
        if score < self.threshold:
            skip_reason = f"graph_score={score:.2f} < threshold={self.threshold}"

        return GraphUtilityScore(score=score, graph_hops=graph_hops, skip_reason=skip_reason)

    def should_use_graph(self, query: str, query_type: str = "") -> bool:
        result = self.estimate(query, query_type)
        return result.score >= self.threshold


_estimator: GraphUtilityEstimator | None = None


def get_graph_utility_estimator() -> GraphUtilityEstimator:
    global _estimator
    if _estimator is None:
        _estimator = GraphUtilityEstimator()
    return _estimator
