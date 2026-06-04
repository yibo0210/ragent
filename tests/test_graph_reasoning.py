import networkx as nx
import pytest
from backend.rag.graph_reasoning.schemas import (
    ReasoningPlan, ReasoningPath, VerificationResult,
    ReasoningStrategy, Verdict,
)
from backend.rag.graph_reasoning.planning import ReasoningPlanner
from backend.rag.graph_reasoning.path_explorer import PathExplorer
from backend.rag.graph_reasoning.path_ranker import PathRanker
from backend.rag.graph_reasoning.subgraph import SubgraphRetriever


class TestReasoningPlan:
    def test_multi_hop_needs_reasoning(self):
        planner = ReasoningPlanner()
        plan = planner.plan("Who founded OpenAI?", "multi_hop")
        assert plan.need_reasoning is True
        assert plan.max_hops == 3

    def test_factoid_skips_reasoning(self):
        planner = ReasoningPlanner()
        plan = planner.plan("What is Redis?", "factoid")
        assert plan.need_reasoning is False

    def test_entity_extraction(self):
        planner = ReasoningPlanner()
        plan = planner.plan("How are OpenAI and Microsoft related?", "entity_relation",
                           entity_names=["OpenAI", "Microsoft"])
        assert "OpenAI" in plan.start_entities
        assert "Microsoft" in plan.start_entities

    def test_auto_entity_extraction(self):
        planner = ReasoningPlanner()
        plan = planner.plan("Who founded OpenAI?", "entity_relation")
        assert "OpenAI" in plan.start_entities


class TestPathExplorer:
    @pytest.fixture
    def sample_graph(self):
        G = nx.DiGraph()
        G.add_edge("OpenAI", "Sam Altman", predicate="CEO", weight=1.0)
        G.add_edge("Sam Altman", "Y Combinator", predicate="WORKED_AT", weight=0.9)
        return G

    def test_explore_finds_paths(self, sample_graph):
        plan = ReasoningPlan(start_entities=["OpenAI"], max_hops=3,
            reasoning_strategy=ReasoningStrategy.MULTI_HOP)
        explorer = PathExplorer()
        paths = explorer.explore(sample_graph, plan)
        assert len(paths) >= 2
        assert any("Y Combinator" in p.nodes for p in paths)

    def test_beam_search(self, sample_graph):
        explorer = PathExplorer()
        paths = explorer.beam_search(sample_graph, "OpenAI", max_hops=3)
        assert len(paths) >= 1

    def test_empty_graph(self):
        explorer = PathExplorer()
        G = nx.DiGraph()
        plan = ReasoningPlan(start_entities=["X"], max_hops=3,
            reasoning_strategy=ReasoningStrategy.MULTI_HOP)
        paths = explorer.explore(G, plan)
        assert paths == []

    def test_missing_start_entity(self, sample_graph):
        plan = ReasoningPlan(start_entities=["Nonexistent"], max_hops=3,
            reasoning_strategy=ReasoningStrategy.MULTI_HOP)
        explorer = PathExplorer()
        paths = explorer.explore(sample_graph, plan)
        assert paths == []


class TestPathRanker:
    def test_ranks_by_hops(self):
        paths = [
            ReasoningPath(nodes=["A", "B", "C"], edges=["R1", "R2"], hop_count=2),
            ReasoningPath(nodes=["A", "D"], edges=["R3"], hop_count=1),
        ]
        ranker = PathRanker()
        ranked = ranker.rank(paths, query="A B C")
        assert ranked[0].path_score >= ranked[-1].path_score

    def test_top_k(self):
        paths = [ReasoningPath(nodes=[f"A{i}"], edges=[], hop_count=i) for i in range(10)]
        ranker = PathRanker()
        top = ranker.top_k(paths, k=3)
        assert len(top) == 3

    def test_empty(self):
        ranker = PathRanker()
        assert ranker.rank([]) == []


class TestSubgraphRetriever:
    def test_empty_entities(self):
        sr = SubgraphRetriever()
        G = sr.retrieve([])
        assert G.number_of_nodes() == 0


class TestSchemas:
    def test_reasoning_path_serialization(self):
        path = ReasoningPath(nodes=["A", "B"], edges=["R1"], hop_count=1, path_score=0.85)
        d = path.model_dump()
        assert d["nodes"] == ["A", "B"]
        assert d["path_score"] == 0.85

    def test_verification_result(self):
        v = VerificationResult(verdict=Verdict.SUPPORTED, confidence=0.9, explanation="OK")
        d = v.model_dump()
        assert d["verdict"] == "SUPPORTED"

    def test_reasoning_plan_defaults(self):
        plan = ReasoningPlan()
        assert plan.need_reasoning is False
        assert plan.max_hops == 3
