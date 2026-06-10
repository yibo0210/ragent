# backend/research/reviewer.py
"""ResearchReviewer: evaluates evidence sufficiency for research goals."""

from __future__ import annotations

from backend.research.schemas import ResearchPlan, ResearchState, ReviewResult


class ResearchReviewer:
    """Reviews collected evidence and determines if research is complete."""

    async def review(self, state: ResearchState, plan: ResearchPlan) -> ReviewResult:
        """Evaluate evidence coverage, diversity, and quality."""
        if not plan:
            return ReviewResult(is_sufficient=True)

        total_tasks = len(plan.tasks)
        if total_tasks == 0:
            return ReviewResult(is_sufficient=True)

        # Coverage: tasks with at least 1 evidence item
        tasks_with_evidence: set[str] = set()
        for ev in state.evidence:
            tasks_with_evidence.add(ev.task_id)
        coverage = len(tasks_with_evidence) / total_tasks

        # Diversity: unique sources
        sources = set(ev.source for ev in state.evidence)
        diversity = min(1.0, len(sources) / 4.0)  # 4 possible sources

        # Citation quality: evidence items with citations
        cited = sum(1 for ev in state.evidence if ev.citation)
        citation_score = cited / max(1, len(state.evidence))

        # Confidence: average confidence level
        confidence_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
        total_conf = sum(confidence_map.get(ev.confidence.value, 0.5) for ev in state.evidence)
        avg_confidence = total_conf / max(1, len(state.evidence))

        # Overall score: weighted composite
        overall = 0.35 * coverage + 0.20 * diversity + 0.25 * citation_score + 0.20 * avg_confidence

        # Identify gaps
        gaps = []
        for task in plan.tasks:
            if task.task_id not in tasks_with_evidence:
                gaps.append(f"No evidence for task: {task.name}")

        recommendations = []
        if coverage < 0.7:
            recommendations.append(f"Evidence coverage is {coverage:.0%}, need more data collection")
        if avg_confidence < 0.6:
            recommendations.append("Average evidence confidence is low, seek higher quality sources")
        if citation_score < 0.5:
            recommendations.append("Many evidence items lack citations, improve source attribution")

        return ReviewResult(
            is_sufficient=overall >= 0.70 and coverage >= 0.60,
            coverage_score=round(coverage, 3),
            diversity_score=round(diversity, 3),
            citation_score=round(citation_score, 3),
            confidence_score=round(avg_confidence, 3),
            overall_score=round(overall, 3),
            gaps=gaps,
            recommendations=recommendations,
        )


_reviewer: ResearchReviewer | None = None


def get_research_reviewer() -> ResearchReviewer:
    global _reviewer
    if _reviewer is None:
        _reviewer = ResearchReviewer()
    return _reviewer
