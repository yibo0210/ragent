# backend/research/gap_analyzer.py
"""GapAnalyzer: identifies knowledge gaps and generates supplementary queries."""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.research.schemas import ResearchState, ReviewResult, GapAnalysis


_GAP_ANALYZER_PROMPT = """You are a research gap analyst. Given the research goal and identified gaps, generate supplementary research queries.

Research Goal: {goal}

Completed Tasks:
{completed_tasks}

Evidence Gaps:
{gaps}

Review Findings:
{recommendations}

Output ONLY valid JSON:
{{
  "gap_analyses": [
    {{
      "task_id": "original task ID with gap",
      "missing_aspect": "what specific information is missing",
      "supplementary_query": "specific search/research query to fill the gap",
      "priority": 0.0-1.0
    }}
  ]
}}

Rules:
- Each gap should map to ONE supplementary query
- Queries should be specific and answerable
- Priority 0.8+: critical gap blocking conclusions
- Priority 0.5-0.7: important but not blocking
- Priority <0.5: nice to have
"""


class GapAnalyzer:
    """Analyzes research gaps and generates supplementary queries for auto-retry."""

    async def analyze(self, state: ResearchState, review: ReviewResult) -> GapAnalysis:
        """Generate supplementary queries from identified gaps."""
        if review.is_sufficient:
            return GapAnalysis()

        # Try LLM-based gap analysis for richer supplementary queries
        try:
            from backend.agent.model_router import get_model_for_agent

            completed = "\n".join(
                f"- [{tid}]: {state.task_results.get(tid, {}).get('finding', 'no result')[:200]}"
                for tid in state.completed_tasks
            )
            gaps_text = "\n".join(f"- {g}" for g in review.gaps)
            recs_text = "\n".join(f"- {r}" for r in review.recommendations)

            goal = state.plan.goal if state.plan else ""

            model = get_model_for_agent("supervisor")
            response = await model.ainvoke([
                SystemMessage(content=_GAP_ANALYZER_PROMPT.format(
                    goal=goal,
                    completed_tasks=completed or "(none)",
                    gaps=gaps_text,
                    recommendations=recs_text,
                )),
                HumanMessage(content="Analyze gaps and generate supplementary queries:"),
            ])

            content = response.content if hasattr(response, "content") else str(response)
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group(0))
                items = data.get("gap_analyses", [])
                if items:
                    # Return the highest-priority gap
                    best = max(items, key=lambda x: x.get("priority", 0))
                    return GapAnalysis(
                        task_id=best.get("task_id", ""),
                        missing_aspect=best.get("missing_aspect", ""),
                        supplementary_query=best.get("supplementary_query", ""),
                        priority=best.get("priority", 0.5),
                    )
        except Exception:
            pass

        # Fallback: heuristic gap analysis
        if review.gaps:
            return GapAnalysis(
                task_id="",
                missing_aspect=review.gaps[0],
                supplementary_query=review.gaps[0],
                priority=0.7,
            )

        return GapAnalysis()


_analyzer: GapAnalyzer | None = None


def get_gap_analyzer() -> GapAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = GapAnalyzer()
    return _analyzer
