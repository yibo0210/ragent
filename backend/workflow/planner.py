"""WorkflowPlanner: decomposes natural language goals into executable DAG plans."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from backend.workflow.schemas import WorkflowStep, WorkflowPlan
from backend.workflow.models import WorkflowDefinition


_PLANNER_SYSTEM_PROMPT = """You are a workflow planning expert. Given a user's business goal, you MUST output a JSON execution plan with these rules:

1. Decompose the goal into concrete steps. Each step invokes exactly one tool.
2. Available tools:
   - rag_specialist: knowledge base search (documents, manuals, reports)
   - web_searcher: real-time web search (news, market data, external info)
   - data_analyst: SQL query for structured data (sales, metrics, logs)
   - local_graph_search: graph entity exploration (who/what is related to X)
   - global_graph_search: community-level summary (high-level topic clusters)
   - direct_answer: simple LLM response (no retrieval needed)

3. For each step, specify:
   - step_id: unique ID (step_1, step_2, ...)
   - name: short human-readable name
   - tool: one of the above tool names
   - query: specific natural language task for this step
   - dependencies: list of step_ids that must complete BEFORE this step
   - input_mapping: if this step depends on prior results, map variable names

4. Dependency rules:
   - Steps with NO dependencies run in PARALLEL
   - Steps with dependencies WAIT for those to finish
   - Chain sequential steps via dependencies

5. Think about what makes sense: data query -> analysis -> visualization -> report

Output ONLY valid JSON:
{
  "steps": [
    {
      "step_id": "step_1",
      "name": "...",
      "tool": "...",
      "query": "...",
      "dependencies": [],
      "input_mapping": {},
      "timeout": 300
    }
  ],
  "reasoning": "why you chose this decomposition"
}
"""


class WorkflowPlanner:
    """Converts natural language goals into WorkflowPlan DAGs."""

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from backend.agent.model_router import get_model_for_agent
            self._model = get_model_for_agent("supervisor")
        return self._model

    async def plan(
        self,
        goal: str,
        tenant_id: int = 0,
        user_id: int = 0,
    ) -> WorkflowPlan:
        """Generate a workflow plan from a user goal."""
        model = self._get_model()

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Goal: {goal}\n\nGenerate a JSON execution plan."),
        ]

        response = await model.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            raise ValueError(f"Planner did not produce valid JSON: {content[:200]}")

        plan_dict = json.loads(json_match.group(0))

        steps = []
        for s in plan_dict.get("steps", []):
            steps.append(WorkflowStep(
                step_id=s["step_id"],
                name=s.get("name", s["step_id"]),
                tool=s.get("tool", "rag_specialist"),
                query=s.get("query", ""),
                dependencies=s.get("dependencies", []),
                input_mapping=s.get("input_mapping", {}),
                timeout=s.get("timeout", 300),
            ))

        estimated_tokens = len(steps) * 500 + 200

        return WorkflowPlan(
            goal=goal,
            steps=steps,
            reasoning=plan_dict.get("reasoning", ""),
            estimated_tokens=estimated_tokens,
        )

    def save_plan(
        self,
        plan: WorkflowPlan,
        tenant_id: int,
        user_id: int,
        db,
        name: str = "",
    ) -> int:
        """Persist a plan to MySQL, returns definition_id."""
        definition = WorkflowDefinition(
            name=name or plan.goal[:100],
            description="",
            goal=plan.goal,
            steps_json=[s.model_dump() for s in plan.steps],
            reasoning=plan.reasoning,
            tenant_id=tenant_id,
            created_by=user_id,
            created_at=datetime.now(timezone.utc),
        )
        db.add(definition)
        db.flush()
        return definition.id

    def load_plan(self, definition_id: int, db) -> WorkflowPlan:
        """Load a persisted plan from MySQL."""
        definition = db.query(WorkflowDefinition).filter(
            WorkflowDefinition.id == definition_id
        ).first()
        if not definition:
            raise ValueError(f"Workflow definition {definition_id} not found")

        steps = [WorkflowStep(**s) for s in (definition.steps_json or [])]
        return WorkflowPlan(
            goal=definition.goal,
            steps=steps,
            reasoning=definition.reasoning or "",
        )


_planner: WorkflowPlanner | None = None


def get_workflow_planner() -> WorkflowPlanner:
    global _planner
    if _planner is None:
        _planner = WorkflowPlanner()
    return _planner
