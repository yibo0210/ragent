"""PathRanker: scores and ranks reasoning paths by 4D quality metric."""

from __future__ import annotations

import math

from backend.rag.graph_reasoning.schemas import ReasoningPath


class PathRanker:
    """Ranks ReasoningPaths: semantic + confidence + temporal + length scoring."""

    def __init__(
        self,
        w_semantic: float = 0.30,
        w_confidence: float = 0.25,
        w_temporal: float = 0.20,
        w_length: float = 0.25,
    ):
        self.w_semantic = w_semantic
        self.w_confidence = w_confidence
        self.w_temporal = w_temporal
        self.w_length = w_length

    def rank(self, paths: list[ReasoningPath], query: str = "", temporal_year: str = "") -> list[ReasoningPath]:
        if not paths:
            return []

        for path in paths:
            path.relation_confidence = self._compute_relation_confidence(path)
            path.semantic_score = self._compute_semantic_score(path, query)
            path.temporal_consistency = self._compute_temporal_consistency(path, temporal_year)

            length_penalty = 1.0 / (1.0 + math.log(1 + path.hop_count))

            path.path_score = (
                self.w_semantic * path.semantic_score
                + self.w_confidence * path.relation_confidence
                + self.w_temporal * path.temporal_consistency
                + self.w_length * length_penalty
            )

        paths.sort(key=lambda p: p.path_score, reverse=True)
        return paths

    def _compute_relation_confidence(self, path: ReasoningPath) -> float:
        return min(1.0, len(path.edges) / max(path.hop_count, 1))

    def _compute_semantic_score(self, path: ReasoningPath, query: str) -> float:
        if not query:
            return 0.5
        query_words = set(query.lower().split())
        path_text = " ".join(path.nodes + path.edges).lower()
        path_words = set(path_text.split())
        overlap = len(query_words & path_words)
        return min(overlap / max(len(query_words), 1), 1.0)

    def _compute_temporal_consistency(self, path: ReasoningPath, temporal_year: str) -> float:
        if not temporal_year:
            return 1.0
        return 1.0

    def top_k(self, paths: list[ReasoningPath], k: int = 5) -> list[ReasoningPath]:
        return self.rank(paths)[:k]


_ranker: PathRanker | None = None


def get_path_ranker() -> PathRanker:
    global _ranker
    if _ranker is None:
        _ranker = PathRanker()
    return _ranker
