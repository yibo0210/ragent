"""WorkflowExecutor: LangGraph-based DAG execution engine.

Supports:
- Serial execution (dependency chains)
- Parallel execution (independent steps via concurrent execution)
- State persistence for resume via MySQL checkpointer
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from backend.workflow.schemas import (
    WorkflowPlan,
    ExecutionStatus,
)
from backend.workflow.tool_runtime import ToolResult, get_tool_registry


class WorkflowGraphState(TypedDict):
    execution_id: str
    plan: dict
    status: str
    current_step_id: str
    completed_steps: list[str]
    step_results: dict[str, dict]
    progress: float
    error_message: str
    user_context: dict


class WorkflowExecutor:
    """Executes a WorkflowPlan as a LangGraph DAG."""

    def __init__(self, checkpointer: BaseCheckpointSaver | None = None):
        self._checkpointer = checkpointer
        self._graph = None

    def _build_graph(self) -> StateGraph:
        if self._graph is not None:
            return self._graph

        builder = StateGraph(WorkflowGraphState)

        builder.add_node("init", self._init_node)
        builder.add_node("execute_step", self._execute_step_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_node("handle_error", self._error_node)

        builder.set_entry_point("init")
        builder.add_edge("init", "execute_step")

        builder.add_conditional_edges(
            "execute_step",
            self._route_after_step,
            {
                "continue": "execute_step",
                "finalize": "finalize",
                "error": "handle_error",
            },
        )

        builder.add_edge("finalize", END)
        builder.add_edge("handle_error", END)

        self._graph = builder.compile(checkpointer=self._checkpointer)
        return self._graph

    async def _init_node(self, state: WorkflowGraphState) -> dict:
        state["status"] = ExecutionStatus.RUNNING.value
        state["completed_steps"] = []
        state["step_results"] = {}
        state["progress"] = 0.0
        state["error_message"] = ""
        return state

    async def _execute_step_node(self, state: WorkflowGraphState) -> dict:
        plan = WorkflowPlan(**state["plan"])
        completed = set(state.get("completed_steps", []))
        total_steps = len(plan.steps)

        ready_steps = []
        for step in plan.steps:
            if step.step_id in completed:
                continue
            deps_satisfied = all(d in completed for d in step.dependencies)
            if deps_satisfied:
                ready_steps.append(step)

        if not ready_steps:
            if len(completed) == total_steps:
                return state
            return {"error_message": "Deadlock: no ready steps but not all completed"}

        registry = get_tool_registry()

        # Execute all ready steps in parallel via asyncio.gather
        async def _run_step(step):
            tool = registry.get(step.tool)
            if tool is None:
                return step, ToolResult(success=False, error=f"Tool not found: {step.tool}")
            previous_results = {
                dep_id: ToolResult(**state["step_results"][dep_id])
                for dep_id in step.dependencies
                if dep_id in state.get("step_results", {})
            }
            try:
                result = await asyncio.wait_for(
                    tool.invoke(
                        query=step.query, user_context=state.get("user_context", {}),
                        step=step, previous_results=previous_results,
                    ),
                    timeout=step.timeout,
                )
                return step, result
            except asyncio.TimeoutError:
                return step, ToolResult(success=False, error=f"Step timed out after {step.timeout}s")
            except Exception as e:
                return step, ToolResult(success=False, error=str(e))

        results = await asyncio.gather(*[_run_step(s) for s in ready_steps], return_exceptions=True)
        for item in results:
            if isinstance(item, tuple):
                step, result = item
                state["step_results"][step.step_id] = result.to_dict()
                state["completed_steps"].append(step.step_id)

        state["progress"] = (len(state["completed_steps"]) / max(total_steps, 1)) * 100.0
        state["current_step_id"] = state["completed_steps"][-1] if state["completed_steps"] else ""
        return state

    def _route_after_step(self, state: WorkflowGraphState) -> str:
        if state.get("error_message"):
            return "error"
        plan = WorkflowPlan(**state["plan"])
        completed = set(state.get("completed_steps", []))
        if len(completed) == len(plan.steps):
            return "finalize"
        return "continue"

    async def _finalize_node(self, state: WorkflowGraphState) -> dict:
        state["status"] = ExecutionStatus.COMPLETED.value
        state["progress"] = 100.0
        return state

    async def _error_node(self, state: WorkflowGraphState) -> dict:
        state["status"] = ExecutionStatus.FAILED.value
        return state

    async def execute(
        self,
        plan: WorkflowPlan,
        execution_id: str,
        user_context: dict,
        session_id: str = "",
    ) -> dict:
        graph = self._build_graph()

        initial_state: WorkflowGraphState = {
            "execution_id": execution_id,
            "plan": plan.model_dump(),
            "status": ExecutionStatus.PENDING.value,
            "current_step_id": "",
            "completed_steps": [],
            "step_results": {},
            "progress": 0.0,
            "error_message": "",
            "user_context": user_context,
        }

        config = {"configurable": {"thread_id": execution_id}}
        final_state = await graph.ainvoke(initial_state, config)
        return final_state


_executor: WorkflowExecutor | None = None


def get_workflow_executor() -> WorkflowExecutor:
    global _executor
    if _executor is None:
        from backend.storage.checkpointer import _get_checkpointer
        _executor = WorkflowExecutor(checkpointer=_get_checkpointer())
    return _executor
