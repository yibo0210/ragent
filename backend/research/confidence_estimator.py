"""ConfidenceEstimator: multi-dimensional evidence confidence scoring."""

from __future__ import annotations

from backend.research.schemas import EvidenceNode


class ConfidenceEstimator:
    """Estimates evidence confidence using source authority + cross-validation."""

    SOURCE_AUTHORITY = {
        "web": 0.5,
        "graph": 0.7,
        "data": 0.8,
        "internal_kb": 0.6,
        "mcp": 0.6,
        "user_upload": 0.9,
    }

    def estimate(self, ev: EvidenceNode, corroborating_count: int = 0, refuting_count: int = 0) -> float:
        """Compute confidence from: source authority 20% + corroboration 40% + refutation 30% + citation 10%."""
        base = self.SOURCE_AUTHORITY.get(ev.source, 0.5)
        corr_boost = min(0.3, corroborating_count * 0.1)
        ref_penalty = min(0.5, refuting_count * 0.15)
        cite_bonus = 0.05 if ev.citation else 0.0
        score = base + corr_boost - ref_penalty + cite_bonus
        return max(0.0, min(1.0, score))

    def batch_estimate(
        self, evidence_nodes: list[EvidenceNode],
        conflict_pairs: set[tuple[str, str]],
    ) -> list[EvidenceNode]:
        """Re-estimate confidence for all evidence considering conflicts."""
        for ev in evidence_nodes:
            refuting = sum(1 for a, b in conflict_pairs if ev.node_id in (a, b))
            ev.confidence = self.estimate(ev, refuting_count=refuting)
        return evidence_nodes


_estimator: ConfidenceEstimator | None = None


def get_confidence_estimator() -> ConfidenceEstimator:
    global _estimator
    if _estimator is None:
        _estimator = ConfidenceEstimator()
    return _estimator
