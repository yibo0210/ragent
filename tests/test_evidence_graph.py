# tests/test_evidence_graph.py
"""Tests for v21 Dynamic Research Agent modules."""

import pytest
from unittest.mock import patch

from backend.research.schemas import (
    EvidenceNode, Hypothesis, HypothesisStatus,
    EvidenceRelationType, ConflictDetection, ExpandedQuestion,
)
from backend.research.evidence_graph import EvidenceGraph, get_evidence_graph
from backend.research.confidence_estimator import ConfidenceEstimator


class TestHypothesisSchema:
    def test_create(self):
        h = Hypothesis(hypothesis_id="H1", statement="测试假设", rationale="测试理由")
        assert h.hypothesis_id == "H1"
        assert h.status == HypothesisStatus.UNVERIFIED

    def test_evidence_refs(self):
        h = Hypothesis(hypothesis_id="H2", statement="test",
                       supporting_evidence=["ev_1", "ev_2"],
                       refuting_evidence=["ev_3"])
        assert len(h.supporting_evidence) == 2
        assert len(h.refuting_evidence) == 1

    def test_status_values(self):
        assert HypothesisStatus.UNVERIFIED.value == "unverified"
        assert HypothesisStatus.SUPPORTED.value == "supported"
        assert HypothesisStatus.REFUTED.value == "refuted"


class TestEvidenceNode:
    def test_create(self):
        ev = EvidenceNode(node_id="ev_001", content="test evidence",
                         source="web", confidence=0.75)
        assert ev.node_id == "ev_001"
        assert ev.confidence == 0.75

    def test_auto_id(self):
        ev = EvidenceNode(content="test")
        assert ev.node_id == ""

    def test_with_hypothesis(self):
        ev = EvidenceNode(content="test", hypothesis_id="H1", task_id="T1", execution_id="rx_001")
        assert ev.hypothesis_id == "H1"
        assert ev.task_id == "T1"


class TestConflictDetection:
    def test_no_conflict(self):
        c = ConflictDetection(evidence_a="ev_1", evidence_b="ev_2",
                              has_conflict=False, conflict_type="none")
        assert not c.has_conflict

    def test_with_conflict(self):
        c = ConflictDetection(evidence_a="ev_1", evidence_b="ev_2",
                              has_conflict=True, conflict_type="factual",
                              explanation="Numbers don't match",
                              resolution="Check SEC filing")
        assert c.has_conflict
        assert c.conflict_type == "factual"
        assert c.resolution == "Check SEC filing"


class TestExpandedQuestion:
    def test_create(self):
        q = ExpandedQuestion(question="Intel AI投资实际金额是多少？",
                            source="conflict", priority=0.9,
                            target_hypothesis="H1")
        assert q.source == "conflict"
        assert q.priority == 0.9

    def test_defaults(self):
        q = ExpandedQuestion(question="test question")
        assert q.source == ""
        assert q.priority == 0.0


class TestEvidenceGraph:
    def test_singleton(self):
        g1 = get_evidence_graph()
        g2 = get_evidence_graph()
        assert g1 is g2

    @patch("backend.research.evidence_graph.write_cypher")
    def test_evidence_node_auto_id(self, mock_write):
        g = EvidenceGraph()
        ev = EvidenceNode(content="test", source="web")
        g.add_evidence(ev)
        assert ev.node_id != ""
        assert ev.node_id.startswith("ev_")


class TestConfidenceEstimator:
    def test_high_authority_source(self):
        ce = ConfidenceEstimator()
        ev = EvidenceNode(content="test", source="data", confidence=0.5)
        score = ce.estimate(ev)
        assert score > 0.5

    def test_corroboration_boost(self):
        ce = ConfidenceEstimator()
        ev = EvidenceNode(content="test", source="web", confidence=0.5)
        s1 = ce.estimate(ev, corroborating_count=0)
        s2 = ce.estimate(ev, corroborating_count=5)
        assert s2 > s1

    def test_refutation_penalty(self):
        ce = ConfidenceEstimator()
        ev = EvidenceNode(content="test", source="web", confidence=0.5)
        s1 = ce.estimate(ev, refuting_count=0)
        s2 = ce.estimate(ev, refuting_count=5)
        assert s2 < s1

    def test_citation_bonus(self):
        ce = ConfidenceEstimator()
        ev1 = EvidenceNode(content="test", source="web", citation="")
        ev2 = EvidenceNode(content="test", source="web", citation="https://example.com")
        assert ce.estimate(ev2) > ce.estimate(ev1)

    def test_batch_estimate(self):
        ce = ConfidenceEstimator()
        evs = [
            EvidenceNode(node_id="ev_1", content="test1", source="web"),
            EvidenceNode(node_id="ev_2", content="test2", source="web"),
        ]
        result = ce.batch_estimate(evs, {("ev_1", "ev_2")})
        assert len(result) == 2
        assert result[0].confidence < 0.5  # penalized by conflict


class TestEvidenceRelationType:
    def test_enum_values(self):
        assert EvidenceRelationType.SUPPORTS.value == "SUPPORTS"
        assert EvidenceRelationType.REFUTES.value == "REFUTES"
        assert EvidenceRelationType.RELATES.value == "RELATES_TO"
