# backend/research/__init__.py

from backend.research.schemas import (
    ResearchPlan, ResearchTask, ResearchTaskStatus,
    Evidence, EvidenceSource, EvidenceConfidence,
    ResearchReport, ReportSection, ReportFormat,
    ResearchState, ReviewResult, GapAnalysis,
)


def _try_import_stub(module_name, names):
    """Lazy import: returns tuple of stubs if module not yet created."""
    try:
        mod = __import__(module_name, fromlist=names)
        return tuple(getattr(mod, n) for n in names)
    except ImportError:
        return tuple(None for _ in names)


(
    ResearchPlanner, get_research_planner,
) = _try_import_stub("backend.research.planner", ["ResearchPlanner", "get_research_planner"])

(
    ResearchExecutor, get_research_executor,
) = _try_import_stub("backend.research.executor", ["ResearchExecutor", "get_research_executor"])

(
    EvidenceStore, get_evidence_store,
) = _try_import_stub("backend.research.evidence_store", ["EvidenceStore", "get_evidence_store"])

(
    ResearchReviewer, get_research_reviewer,
) = _try_import_stub("backend.research.reviewer", ["ResearchReviewer", "get_research_reviewer"])

(
    GapAnalyzer, get_gap_analyzer,
) = _try_import_stub("backend.research.gap_analyzer", ["GapAnalyzer", "get_gap_analyzer"])

(
    ResearchReportGenerator, get_report_generator,
) = _try_import_stub("backend.research.report_generator", ["ResearchReportGenerator", "get_report_generator"])


__all__ = [
    "ResearchPlan", "ResearchTask", "ResearchTaskStatus",
    "Evidence", "EvidenceSource", "EvidenceConfidence",
    "ResearchReport", "ReportSection", "ReportFormat",
    "ResearchState", "ReviewResult", "GapAnalysis",
    "ResearchPlanner", "get_research_planner",
    "ResearchExecutor", "get_research_executor",
    "EvidenceStore", "get_evidence_store",
    "ResearchReviewer", "get_research_reviewer",
    "GapAnalyzer", "get_gap_analyzer",
    "ResearchReportGenerator", "get_report_generator",
]
