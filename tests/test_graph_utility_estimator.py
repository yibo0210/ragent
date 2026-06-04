import pytest
from backend.rag.graph_utility_estimator import (
    GraphUtilityEstimator,
    GraphUtilityScore,
    get_graph_utility_estimator,
)


class TestGraphUtilityScore:
    def test_create(self):
        s = GraphUtilityScore(score=0.75, graph_hops=3, skip_reason="")
        assert s.score == 0.75
        assert s.graph_hops == 3
        assert s.skip_reason == ""

    def test_below_threshold(self):
        s = GraphUtilityScore(score=0.2, graph_hops=0, skip_reason="graph_score=0.20 < threshold=0.35")
        assert s.score < 0.35


class TestGraphUtilityEstimator:
    @pytest.fixture
    def estimator(self):
        return GraphUtilityEstimator(threshold=0.35)

    def test_factoid_query_low_score(self, estimator):
        score = estimator.estimate("What is Redis?", "factoid")
        assert score.score < 0.5, f"Expected low score for factoid, got {score.score:.2f}"

    def test_multi_hop_query_higher_score(self, estimator):
        score = estimator.estimate(
            "Trace the investment chain from SoftBank to ByteDance.",
            "multi_hop",
        )
        assert score.score > 0.15, f"Expected higher score for multi-hop, got {score.score:.2f}"

    def test_entity_relation_score(self, estimator):
        score = estimator.estimate("Who founded OpenAI?", "entity_relation")
        assert score.score > 0.2, f"Expected moderate score for entity relation, got {score.score:.2f}"

    def test_should_use_graph_false_for_factoid(self, estimator):
        assert estimator.should_use_graph("What is Redis?", "factoid") is False

    def test_singleton(self):
        e1 = get_graph_utility_estimator()
        e2 = get_graph_utility_estimator()
        assert e1 is e2
