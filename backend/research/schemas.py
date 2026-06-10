# backend/research/schemas.py
"""Research domain schemas: Plan, Task, Evidence, Report."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ResearchTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvidenceSource(str, Enum):
    GRAPH_RAG = "graph_rag"
    WEB_SEARCH = "web_search"
    DATA_ANALYST = "data_analyst"
    INTERNAL_KB = "internal_kb"
    MCP = "mcp"
    USER_UPLOAD = "user_upload"


class EvidenceConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Evidence(BaseModel):
    """Single piece of evidence collected during research."""

    id: str = ""
    task_id: str = ""
    source: EvidenceSource = EvidenceSource.WEB_SEARCH
    content: str = ""
    citation: str = ""
    confidence: EvidenceConfidence = EvidenceConfidence.MEDIUM
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResearchTask(BaseModel):
    """A single task in the research plan DAG."""

    task_id: str = Field(..., description="Unique task ID, e.g. 'T1'")
    name: str = Field(..., description="Human-readable task name")
    description: str = ""
    agent: str = Field(..., description="Agent: web|graph|data|internal_kb")
    query: str = Field(..., description="Research question for this task")
    dependencies: list[str] = Field(default_factory=list)
    status: ResearchTaskStatus = ResearchTaskStatus.PENDING
    evidence_ids: list[str] = Field(default_factory=list)
    timeout: int = 60


class ResearchPlan(BaseModel):
    """Full research execution plan DAG."""

    plan_id: str = ""
    goal: str = ""
    tasks: list[ResearchTask] = Field(default_factory=list)
    reasoning: str = ""
    estimated_duration_minutes: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResearchState(BaseModel):
    """Runtime state of a research execution."""

    execution_id: str = ""
    plan: Optional[ResearchPlan] = None
    status: ResearchTaskStatus = ResearchTaskStatus.PENDING
    current_task_id: str = ""
    completed_tasks: list[str] = Field(default_factory=list)
    task_results: dict[str, dict] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    review_count: int = 0
    max_review_rounds: int = 3
    gap_analyses: list[GapAnalysis] = Field(default_factory=list)
    # v21: Dynamic Research fields
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    evidence_graph: list[EvidenceNode] = Field(default_factory=list)
    conflicts: list[ConflictDetection] = Field(default_factory=list)
    expanded_questions: list[ExpandedQuestion] = Field(default_factory=list)
    dynamic_round: int = 0
    max_dynamic_rounds: int = 2
    progress: float = 0.0
    error_message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class ReviewResult(BaseModel):
    """Output of the evidence review phase."""

    is_sufficient: bool = False
    coverage_score: float = 0.0
    diversity_score: float = 0.0
    citation_score: float = 0.0
    confidence_score: float = 0.0
    overall_score: float = 0.0
    gaps: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class GapAnalysis(BaseModel):
    """Identifies missing evidence and generates supplementary queries."""

    task_id: str = ""
    missing_aspect: str = ""
    supplementary_query: str = ""
    priority: float = 0.0


class HypothesisStatus(str, Enum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"


class Hypothesis(BaseModel):
    hypothesis_id: str = Field(..., description="e.g. 'H1'")
    statement: str = Field(..., description="假设陈述")
    rationale: str = ""
    status: HypothesisStatus = HypothesisStatus.UNVERIFIED
    supporting_evidence: list[str] = Field(default_factory=list)
    refuting_evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    verification_tasks: list[str] = Field(default_factory=list)


class EvidenceRelationType(str, Enum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    RELATES = "RELATES_TO"


class EvidenceNode(BaseModel):
    node_id: str = ""
    content: str = ""
    source: str = ""
    citation: str = ""
    confidence: float = 0.5
    hypothesis_id: str = ""
    task_id: str = ""
    execution_id: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConflictDetection(BaseModel):
    evidence_a: str = ""
    evidence_b: str = ""
    has_conflict: bool = False
    conflict_type: str = ""
    explanation: str = ""
    resolution: str = ""


class ExpandedQuestion(BaseModel):
    question: str = ""
    source: str = ""
    priority: float = 0.0
    target_hypothesis: str = ""


class ReportFormat(str, Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    PPTX = "pptx"


class ReportSection(BaseModel):
    """A section of the research report."""

    heading: str = ""
    content: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    subsections: list[ReportSection] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Complete research report with evidence bindings."""

    report_id: str = ""
    execution_id: str = ""
    title: str = ""
    executive_summary: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    evidence_map: dict[str, str] = Field(default_factory=dict)
    confidence_summary: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Resolve forward reference for ResearchState.gap_analyses
ResearchState.model_rebuild()
