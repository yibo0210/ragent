# backend/research/planner.py
"""ResearchPlanner: decomposes research goals into DAG execution plans."""

from __future__ import annotations

import json
import re
import uuid

from langchain_core.messages import SystemMessage, HumanMessage

from backend.research.schemas import ResearchPlan, ResearchTask


_RESEARCH_PLANNER_PROMPT = """Decompose a research goal into a DAG of tasks. Output ONLY JSON.

Agents: web (search), graph (relationships), data (SQL), internal_kb (documents)

JSON format:
{
  "tasks": [
    {"task_id": "T1", "name": "...", "description": "...", "agent": "web|graph|data|internal_kb", "query": "...", "dependencies": [], "timeout": 60}
  ],
  "reasoning": "..."
}

Rules: no-dependency tasks run in parallel; chain sequential tasks via dependencies; 3-6 tasks total.
"""


class ResearchPlanner:
    """Converts a research goal into a ResearchPlan with DAG dependencies."""

    async def plan(self, goal: str) -> ResearchPlan:
        from langchain.chat_models import init_chat_model
        from backend.config import get_settings
        settings = get_settings()

        model = init_chat_model(
            model="qwen-turbo",
            model_provider="openai",
            api_key=settings.ark_api_key,
            base_url=settings.base_url,
            temperature=0.0,
            max_tokens=1024,
            timeout=60,
        )
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
