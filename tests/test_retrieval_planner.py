import pytest
from backend.rag.retrieval_planner import RetrievalPlanner, STRATEGY_MAP, RetrievalPlan


class TestRetrievalPlanner:
    def test_all_six_types_defined(self):
        for qt in ["factoid", "entity_relation", "multi_hop", "global_summary", "temporal", "comparison"]:
            assert qt in STRATEGY_MAP, f"{qt} missing from STRATEGY_MAP"

    def test_factoid_skips_graph(self):
        plan = RetrievalPlanner().plan_from_query_type("factoid")
        assert plan.use_graph is False
        assert plan.use_community is False
        assert plan.graph_hops == 0
        assert "dense" in plan.enabled_channels
        assert "graph" not in plan.enabled_channels

    def test_multi_hop_uses_graph(self):
        plan = RetrievalPlanner().plan_from_query_type("multi_hop")
        assert plan.use_graph is True
        assert plan.fusion_strategy == "graph_first"
        assert plan.graph_hops == 3

    def test_global_summary_uses_community(self):
        plan = RetrievalPlanner().plan_from_query_type("global_summary")
        assert plan.use_community is True
        assert plan.use_graph is False

    def test_plan_with_intent_override(self):
        plan = RetrievalPlanner().plan(intent={"query_type": "factoid", "graph_hops": 1})
        assert plan.graph_hops == 1
        assert plan.query_type == "factoid"

    def test_plan_with_graph_skip(self):
        plan = RetrievalPlanner().plan(intent={"query_type": "multi_hop", "graph_skip": True})
        assert plan.use_graph is False
        assert plan.use_community is False

    def test_unknown_type_falls_back_to_factoid(self):
        plan = RetrievalPlanner().plan_from_query_type("nonexistent")
        assert plan.query_type == "factoid"

    def test_enabled_channels(self):
        plan = RetrievalPlanner().plan_from_query_type("comparison")
        channels = plan.enabled_channels
        assert "dense" in channels
        assert "graph" in channels
        assert "community" not in channels
