"""Tests for WorkflowPlanner."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.workflow.planner import WorkflowPlanner
from backend.workflow.schemas import WorkflowPlan, WorkflowStep


class TestWorkflowPlanner:
    def test_plan_parses_valid_json(self):
        async def _test():
            planner = WorkflowPlanner()
            mock_model = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = """{
                "steps": [
                    {
                        "step_id": "step_1",
                        "name": "Query sales data",
                        "tool": "data_analyst",
                        "query": "SELECT total FROM sales WHERE quarter=2",
                        "dependencies": [],
                        "input_mapping": {},
                        "timeout": 300
                    },
                    {
                        "step_id": "step_2",
                        "name": "Generate chart",
                        "tool": "direct_answer",
                        "query": "Create a bar chart of Q2 sales",
                        "dependencies": ["step_1"],
                        "input_mapping": {"step_1": "sales_data"},
                        "timeout": 300
                    }
                ],
                "reasoning": "First query data, then visualize"
            }"""
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            planner._model = mock_model

            plan = await planner.plan("Analyze Q2 sales")

            assert isinstance(plan, WorkflowPlan)
            assert plan.goal == "Analyze Q2 sales"
            assert len(plan.steps) == 2
            assert plan.steps[0].tool == "data_analyst"
            assert plan.steps[0].dependencies == []
            assert plan.steps[1].dependencies == ["step_1"]
            assert plan.reasoning == "First query data, then visualize"
            assert plan.estimated_tokens > 0
        import asyncio
        asyncio.run(_test())

    def test_plan_invalid_json_raises(self):
        async def _test():
            planner = WorkflowPlanner()
            mock_model = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "I cannot generate a plan for this."
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            planner._model = mock_model

            with pytest.raises(ValueError):
                await planner.plan("Invalid goal")
        import asyncio
        asyncio.run(_test())

    def test_save_and_load_plan(self):
        from backend.storage.database import SessionLocal
        from backend.workflow.models import WorkflowDefinition

        planner = WorkflowPlanner()
        plan = WorkflowPlan(
            goal="Test goal",
            steps=[
                WorkflowStep(step_id="step_1", name="Query", tool="data_analyst", query="SELECT 1"),
            ],
            reasoning="Test plan",
        )

        db = SessionLocal()
        try:
            def_id = planner.save_plan(plan, tenant_id=1, user_id=1, db=db)
            db.commit()

            loaded = planner.load_plan(def_id, db)
            assert loaded.goal == "Test goal"
            assert len(loaded.steps) == 1
            assert loaded.steps[0].tool == "data_analyst"

            # Cleanup
            db.query(WorkflowDefinition).filter(WorkflowDefinition.id == def_id).delete()
            db.commit()
        finally:
            db.close()
