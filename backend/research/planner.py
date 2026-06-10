# backend/research/planner.py
"""ResearchPlanner: decomposes research goals into DAG execution plans."""

from __future__ import annotations

import json
import re
import uuid

from langchain_core.messages import SystemMessage, HumanMessage

from backend.research.schemas import ResearchPlan, ResearchTask


_RESEARCH_PLANNER_PROMPT = """You are a research planning expert. Given a research goal, decompose it into a structured investigation plan.

Available research agents:
- web: Web search for industry data, news, reports, market trends
- graph: Knowledge graph exploration for entity relationships, multi-hop reasoning
- data: SQL analysis for structured data, metrics, KPIs
- internal_kb: Internal knowledge base search (documents, manuals, past research)

Output ONLY valid JSON:
{
  "tasks": [
    {
      "task_id": "T1",
      "name": "short task name",
      "description": "what this task investigates",
      "agent": "web|graph|data|internal_kb",
      "query": "specific research question for the agent",
      "dependencies": [],
      "timeout": 60
    }
  ],
  "reasoning": "brief explanation of the plan structure"
}

Rules:
1. First tasks collect broad information (web search, internal KB) — NO dependencies
2. Later tasks analyze and cross-reference (graph reasoning, data analysis) — depend on earlier results
3. Final tasks synthesize and validate — depend on mid-stage results
4. Tasks with NO dependencies run in PARALLEL
5. Each task MUST target exactly ONE agent
6. task_id format: T1, T2, T3, ...
7. 3-8 tasks total depending on goal complexity
"""


class ResearchPlanner:
    """Converts a research goal into a ResearchPlan with DAG dependencies."""

    async def plan(self, goal: str) -> ResearchPlan:
        from backend.agent.model_router import get_model_for_agent

        model = get_model_for_agent("supervisor")
        response = await model.ainvoke([
            SystemMessage(content=_RESEARCH_PLANNER_PROMPT),
            HumanMessage(content=f"Research goal: {goal}\n\nGenerate research plan:"),
        ])

        content = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return ResearchPlan(goal=goal, reasoning="Failed to parse plan")

        data = json.loads(json_match.group(0))
        tasks = []
        for item in data.get("tasks", []):
            tasks.append(ResearchTask(
                task_id=item["task_id"],
                name=item.get("name", ""),
                description=item.get("description", ""),
                agent=item.get("agent", "web"),
                query=item.get("query", ""),
                dependencies=item.get("dependencies", []),
                timeout=item.get("timeout", 600),
            ))

        duration = len(tasks) * 2  # rough estimate: 2 min/task
        return ResearchPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:12]}",
            goal=goal,
            tasks=tasks,
            reasoning=data.get("reasoning", ""),
            estimated_duration_minutes=duration,
        )


_planner: ResearchPlanner | None = None


def get_research_planner() -> ResearchPlanner:
    global _planner
    if _planner is None:
        _planner = ResearchPlanner()
    return _planner
