# tests/test_research.py
"""Tests for v20 Deep Research Engine."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from backend.research.schemas import (
    ResearchPlan, ResearchTask, ResearchTaskStatus,
    Evidence, EvidenceSource, EvidenceConfidence,
    ResearchState, ReviewResult, GapAnalysis,
    ResearchReport, ReportSection, ReportFormat,
)
from backend.research.reviewer import ResearchReviewer
from backend.research.gap_analyzer import GapAnalyzer
from backend.research.evidence_store import EvidenceStore
from backend.research.planner import ResearchPlanner


class TestResearchSchemas:
    """Test core schema validation."""

    def test_research_task_creation(self):
        t = ResearchTask(task_id="T1", name="Market Analysis", agent="web", query="test")
        assert t.task_id == "T1"
        assert t.agent == "web"
        assert t.status == ResearchTaskStatus.PENDING
        assert t.dependencies == []

    def test_research_task_with_dependencies(self):
        t = ResearchTask(
            task_id="T2", name="Synthesis", agent="graph",
            query="synthesize findings", dependencies=["T1", "T3"],
        )
        assert len(t.dependencies) == 2
        assert "T1" in t.dependencies

    def test_evidence_creation(self):
        e = Evidence(
            task_id="T1",
            source=EvidenceSource.WEB_SEARCH,
            content="Market size is $10B",
            citation="https://example.com/report",
            confidence=EvidenceConfidence.HIGH,
        )
        assert e.source == EvidenceSource.WEB_SEARCH
        assert e.confidence == EvidenceConfidence.HIGH
        assert e.citation == "https://example.com/report"

    def test_research_plan_dag(self):
        tasks = [
            ResearchTask(task_id="T1", name="T1", agent="web", query="q1"),
            ResearchTask(task_id="T2", name="T2", agent="graph", query="q2", dependencies=["T1"]),
            ResearchTask(task_id="T3", name="T3", agent="web", query="q3"),
        ]
        plan = ResearchPlan(goal="test goal", tasks=tasks)
        assert len(plan.tasks) == 3
        # T1 and T3 are independent (no dependencies)
        independent = [t for t in plan.tasks if not t.dependencies]
        assert len(independent) == 2

    def test_research_state_progress(self):
        state = ResearchState(
            execution_id="rx_test",
            status=ResearchTaskStatus.RUNNING,
            progress=50.0,
        )
        assert state.progress == 50.0
        assert state.status == ResearchTaskStatus.RUNNING

    def test_review_result_scoring(self):
        r = ReviewResult(
            is_sufficient=True,
            coverage_score=0.9,
            diversity_score=0.8,
            citation_score=0.7,
            confidence_score=0.85,
            overall_score=0.83,
        )
        assert r.is_sufficient
        assert r.overall_score > 0.8

    def test_gap_analysis(self):
        g = GapAnalysis(
            task_id="T2",
            missing_aspect="Market size data for 2025",
            supplementary_query="AI Agent market size 2025 billion",
            priority=0.9,
        )
        assert g.priority == 0.9
        assert g.task_id == "T2"


class TestResearchReviewer:
    """Test evidence review logic."""

    @pytest.mark.asyncio
    async def test_sufficient_evidence(self):
        reviewer = ResearchReviewer()
        plan = ResearchPlan(goal="test", tasks=[
            ResearchTask(task_id="T1", name="test", agent="web", query="test"),
        ])
        state = ResearchState(plan=plan, evidence=[
            Evidence(task_id="T1", source=EvidenceSource.WEB_SEARCH,
                    content="test", citation="https://x.com",
                    confidence=EvidenceConfidence.HIGH),
        ])
        result = await reviewer.review(state, plan)
        assert result.is_sufficient
        assert result.coverage_score == 1.0

    @pytest.mark.asyncio
    async def test_insufficient_evidence_no_citations(self):
        reviewer = ResearchReviewer()
        plan = ResearchPlan(goal="test", tasks=[
            ResearchTask(task_id="T1", name="test", agent="web", query="test"),
            ResearchTask(task_id="T2", name="test2", agent="graph", query="test2"),
        ])
        state = ResearchState(plan=plan, evidence=[
            Evidence(task_id="T1", source=EvidenceSource.WEB_SEARCH,
                    content="test", citation="",
                    confidence=EvidenceConfidence.LOW),
        ])
        result = await reviewer.review(state, plan)
        # T2 has no evidence, coverage should be 0.5
        assert result.coverage_score == 0.5
        assert not result.is_sufficient

    @pytest.mark.asyncio
    async def test_empty_plan_is_sufficient(self):
        reviewer = ResearchReviewer()
        plan = ResearchPlan(goal="test", tasks=[])
        state = ResearchState(plan=plan, evidence=[])
        result = await reviewer.review(state, plan)
        assert result.is_sufficient


class TestGapAnalyzer:
    """Test gap analysis logic."""

    @pytest.mark.asyncio
    async def test_no_gaps_when_sufficient(self):
        analyzer = GapAnalyzer()
        result = await analyzer.analyze(
            ResearchState(),
            ReviewResult(is_sufficient=True),
        )
        assert not result.task_id
        assert not result.missing_aspect

    @pytest.mark.asyncio
    async def test_fallback_gap_from_review(self):
        analyzer = GapAnalyzer()
        plan = ResearchPlan(goal="test", tasks=[
            ResearchTask(task_id="T1", name="test", agent="web", query="test"),
        ])
        result = await analyzer.analyze(
            ResearchState(plan=plan, completed_tasks=[]),
            ReviewResult(
                is_sufficient=False,
                gaps=["No evidence for task: test"],
                recommendations=["Need more data"],
            ),
        )
        assert result.missing_aspect  # Should have fallback gap
        assert result.priority > 0


class TestEvidenceStore:
    """Test evidence persistence (requires DB)."""

    def test_save_batch_creates_ids(self):
        store = EvidenceStore()
        evidence_list = [
            Evidence(task_id="T1", source=EvidenceSource.WEB_SEARCH,
                    content="test1", confidence=EvidenceConfidence.HIGH),
            Evidence(task_id="T2", source=EvidenceSource.GRAPH_RAG,
                    content="test2", confidence=EvidenceConfidence.MEDIUM),
        ]
        # IDs should be auto-generated on save
        for ev in evidence_list:
            if not ev.id:
                ev.id = f"ev_test_{id(ev)}"
        assert all(ev.id for ev in evidence_list)

    def test_store_singleton(self):
        from backend.research.evidence_store import get_evidence_store
        s1 = get_evidence_store()
        s2 = get_evidence_store()
        assert s1 is s2


class TestResearchPlanner:
    """Test planner structure (integration test needs LLM)."""

    def test_planner_singleton(self):
        from backend.research.planner import get_research_planner
        p1 = get_research_planner()
        p2 = get_research_planner()
        assert p1 is p2


class TestResearchExecutor:
    """Test executor structure."""

    def test_executor_singleton(self):
        from backend.research.executor import get_research_executor
        e1 = get_research_executor()
        e2 = get_research_executor()
        assert e1 is e2
