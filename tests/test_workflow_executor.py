"""Tests for WorkflowExecutor."""

import asyncio
import pytest

from backend.workflow.schemas import WorkflowPlan, WorkflowStep, ExecutionStatus
from backend.workflow.executor import WorkflowExecutor
from backend.workflow.tool_runtime import ToolResult, ToolRegistry, WorkflowTool
import backend.workflow.tool_runtime as rt_mod


class TestWorkflowExecutor:
    @pytest.fixture
    def sample_plan(self):
        return WorkflowPlan(
            goal="Test workflow",
            steps=[
                WorkflowStep(
                    step_id="step_1", name="First step", tool="echo",
                    query="echo hello", dependencies=[], input_mapping={}, timeout=10,
                ),
                WorkflowStep(
                    step_id="step_2", name="Second step", tool="echo",
                    query="echo world", dependencies=["step_1"],
                    input_mapping={"step_1": "prev"}, timeout=10,
                ),
            ],
            reasoning="Test plan",
        )

    @pytest.fixture
    def echo_registry(self):
        async def echo_fn(query, user_context=None, step=None, previous_results=None):
            return ToolResult(success=True, data={"echo": query})
        registry = ToolRegistry()
        registry.register(WorkflowTool(name="echo", description="Echo", invoke_fn=echo_fn))
        return registry

    @pytest.fixture
    def install_registry(self, echo_registry):
        rt_mod._tool_registry = echo_registry
        yield
        rt_mod._tool_registry = None

    def test_execute_sequential_plan(self, sample_plan, install_registry):
        async def _test():
            executor = WorkflowExecutor()
            state = await executor.execute(
                plan=sample_plan,
                execution_id="test_exec_seq",
                user_context={"tenant_id": 1, "user_id": 1},
            )
            assert state["status"] == ExecutionStatus.COMPLETED.value
            assert state["progress"] == 100.0
            assert "step_1" in state["step_results"]
            assert "step_2" in state["step_results"]
            assert state["step_results"]["step_1"]["success"] is True
        asyncio.run(_test())

    def test_execute_parallel_plan(self):
        async def _test():
            async def slow_echo(query, **kwargs):
                await asyncio.sleep(0.05)
                return ToolResult(success=True, data={"echo": query})

            registry = ToolRegistry()
            registry.register(WorkflowTool(name="echo", description="d", invoke_fn=slow_echo))
            rt_mod._tool_registry = registry

            plan = WorkflowPlan(
                goal="Test parallel",
                steps=[
                    WorkflowStep(step_id="step_a", name="A", tool="echo", query="task_a"),
                    WorkflowStep(step_id="step_b", name="B", tool="echo", query="task_b"),
                ],
                reasoning="Test",
            )
            try:
                executor = WorkflowExecutor()
                state = await executor.execute(plan, "test_exec_par", {"tenant_id": 1})
                assert state["status"] == ExecutionStatus.COMPLETED.value
                assert "step_a" in state["step_results"]
                assert "step_b" in state["step_results"]
            finally:
                rt_mod._tool_registry = None
        asyncio.run(_test())

    def test_execute_with_failing_step(self, sample_plan):
        async def _test():
            async def fail_fn(query, **kwargs):
                return ToolResult(success=False, error="step failed")

            registry = ToolRegistry()
            registry.register(WorkflowTool(name="echo", description="d", invoke_fn=fail_fn))
            rt_mod._tool_registry = registry

            try:
                executor = WorkflowExecutor()
                state = await executor.execute(sample_plan, "test_exec_fail", {"tenant_id": 1})
                assert "step_1" in state["step_results"]
                assert state["step_results"]["step_1"]["success"] is False
                assert state["step_results"]["step_1"]["error"] == "step failed"
            finally:
                rt_mod._tool_registry = None
        asyncio.run(_test())

    def test_route_after_step_all_complete(self, sample_plan):
        executor = WorkflowExecutor()
        state = {
            "execution_id": "test",
            "plan": sample_plan.model_dump(),
            "status": ExecutionStatus.RUNNING.value,
            "current_step_id": "step_2",
            "completed_steps": ["step_1", "step_2"],
            "step_results": {},
            "progress": 100.0,
            "error_message": "",
            "user_context": {},
        }
        assert executor._route_after_step(state) == "finalize"

    def test_route_after_step_with_error(self, sample_plan):
        executor = WorkflowExecutor()
        state = {
            "execution_id": "test",
            "plan": sample_plan.model_dump(),
            "status": ExecutionStatus.RUNNING.value,
            "current_step_id": "step_1",
            "completed_steps": [],
            "step_results": {},
            "progress": 0.0,
            "error_message": "Deadlock detected",
            "user_context": {},
        }
        assert executor._route_after_step(state) == "error"

    def test_route_after_step_continue(self, sample_plan):
        executor = WorkflowExecutor()
        state = {
            "execution_id": "test",
            "plan": sample_plan.model_dump(),
            "status": ExecutionStatus.RUNNING.value,
            "current_step_id": "step_1",
            "completed_steps": ["step_1"],
            "step_results": {
                "step_1": {"success": True, "data": {}, "error": "", "tokens_used": 0}
            },
            "progress": 50.0,
            "error_message": "",
            "user_context": {},
        }
        assert executor._route_after_step(state) == "continue"
