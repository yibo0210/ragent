"""Workflow API endpoints.

POST /workflows/plan       — Generate a workflow plan from a goal
POST /workflows/execute    — Execute a workflow plan
GET  /workflows/{id}/status — Query execution status
GET  /workflows/{id}/artifacts — List generated artifacts
GET  /workflows            — List user's workflow executions
DELETE /workflows/{id}     — Cancel a running workflow
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from backend.auth.dependencies import UserContext, get_current_user
from backend.storage.database import SessionLocal
from backend.workflow.schemas import (
    WorkflowPlanRequest,
    WorkflowPlanResponse,
    WorkflowExecuteRequest,
    WorkflowExecuteResponse,
    WorkflowStatusResponse,
    WorkflowListResponse,
    WorkflowArtifactRef,
    WorkflowPlan,
    WorkflowStep,
    ExecutionStatus,
    ArtifactType,
)
from backend.workflow.planner import get_workflow_planner
from backend.workflow.models import WorkflowDefinition, WorkflowExecution, WorkflowArtifact
from backend.workflow.executor import get_workflow_executor
from backend.workflow.agent_tools import register_agent_tools

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post("/plan", response_model=WorkflowPlanResponse)
async def plan_workflow(
    request: WorkflowPlanRequest,
    user: UserContext = Depends(get_current_user),
):
    """Generate a workflow execution plan from a natural language goal."""
    planner = get_workflow_planner()
    plan = await planner.plan(
        goal=request.goal,
        tenant_id=user.tenant_id,
        user_id=user.user_id,
    )

    db = SessionLocal()
    try:
        definition_id = planner.save_plan(plan, user.tenant_id, user.user_id, db)
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save workflow plan")
    finally:
        db.close()

    return WorkflowPlanResponse(definition_id=definition_id, plan=plan)


@router.post("/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(
    request: WorkflowExecuteRequest,
    user: UserContext = Depends(get_current_user),
):
    """Start executing a workflow plan."""
    db = SessionLocal()
    try:
        definition = db.query(WorkflowDefinition).filter(
            WorkflowDefinition.id == request.definition_id,
            WorkflowDefinition.tenant_id == user.tenant_id,
        ).first()
        if not definition:
            raise HTTPException(status_code=404, detail="Workflow definition not found")

        steps = [WorkflowStep(**s) for s in (definition.steps_json or [])]
        plan = WorkflowPlan(
            goal=definition.goal,
            steps=steps,
            reasoning=definition.reasoning or "",
        )
    finally:
        db.close()

    execution_id = f"wf_{uuid.uuid4().hex[:12]}"

    register_agent_tools()

    user_context = {
        "user_id": user.user_id,
        "tenant_id": user.tenant_id,
        "tenant_name": user.tenant_name,
        "role": user.role,
        "access_level": user.access_level,
    }

    executor = get_workflow_executor()

    db = SessionLocal()
    try:
        execution = WorkflowExecution(
            execution_id=execution_id,
            definition_id=definition.id,
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            session_id=request.session_id,
            status=ExecutionStatus.RUNNING.value,
        )
        db.add(execution)
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create execution record")
    finally:
        db.close()

    import asyncio
    asyncio.create_task(
        _run_workflow_background(
            execution_id=execution_id,
            plan=plan,
            user_context=user_context,
            session_id=request.session_id or "",
            definition_id=definition.id,
        )
    )

    return WorkflowExecuteResponse(
        execution_id=execution_id,
        status=ExecutionStatus.RUNNING,
    )


async def _run_workflow_background(
    execution_id: str,
    plan,
    user_context: dict,
    session_id: str,
    definition_id: int,
):
    """Background task: execute workflow, generate artifacts, update DB."""
    executor = get_workflow_executor()
    db = SessionLocal()
    try:
        final_state = await executor.execute(
            plan=plan,
            execution_id=execution_id,
            user_context=user_context,
            session_id=session_id,
        )

        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.execution_id == execution_id
        ).first()
        if not execution:
            return

        execution.status = final_state.get("status", ExecutionStatus.COMPLETED.value)
        execution.progress = final_state.get("progress", 100.0)
        execution.completed_at = datetime.now(timezone.utc)
        execution.state_json = final_state

        # Generate artifacts from step results
        step_results = final_state.get("step_results", {})
        if step_results:
            try:
                from backend.workflow.artifact import get_artifact_generator
                gen = get_artifact_generator()
                report = await gen.generate_report(
                    title=plan.goal,
                    step_results=step_results,
                    user_context=user_context,
                )
                db.add(WorkflowArtifact(
                    execution_id=execution.id,
                    step_id="synthesize",
                    artifact_type=report.artifact_type.value,
                    title=report.title or plan.goal,
                    mime_type=report.mime_type,
                    content=report.content,
                ))
                # Extract structured data from data_analyst results for CSV/Excel
                for step_id, result in step_results.items():
                    if result.get("success") and result.get("data"):
                        data = result["data"]
                        if isinstance(data, dict) and "response" in data:
                            db.add(WorkflowArtifact(
                                execution_id=execution.id,
                                step_id=step_id,
                                artifact_type="report",
                                title=f"Step: {step_id}",
                                mime_type="text/plain",
                                content=str(data["response"])[:10000],
                            ))
            except Exception as e:
                from backend.observability import get_logger
                get_logger("ragent.workflow").warning(
                    "artifact_generation_failed", error=str(e)
                )

        db.commit()

    except Exception as e:
        db.rollback()
        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.execution_id == execution_id
        ).first()
        if execution:
            execution.status = ExecutionStatus.FAILED.value
            execution.error_message = str(e)
            db.commit()
    finally:
        db.close()


@router.get("/{execution_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    execution_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Get the current status of a workflow execution."""
    db = SessionLocal()
    try:
        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.execution_id == execution_id,
            WorkflowExecution.tenant_id == user.tenant_id,
        ).first()
        if not execution:
            raise HTTPException(status_code=404, detail="Workflow execution not found")

        artifacts = [
            WorkflowArtifactRef(
                artifact_id=a.id,
                step_id=a.step_id,
                artifact_type=ArtifactType(a.artifact_type),
                title=a.title,
                mime_type=a.mime_type or "text/markdown",
            )
            for a in (execution.artifacts or [])
        ]

        state = execution.state_json or {}
        return WorkflowStatusResponse(
            execution_id=execution.execution_id,
            status=ExecutionStatus(execution.status),
            progress=execution.progress or 0,
            current_step_id=execution.current_step_id,
            step_results=state.get("step_results", {}),
            artifacts=artifacts,
            error_message=execution.error_message,
            goal=(execution.definition.goal if execution.definition else ""),
        )
    finally:
        db.close()


@router.get("/{execution_id}/artifacts")
async def get_workflow_artifacts(
    execution_id: str,
    user: UserContext = Depends(get_current_user),
):
    """List all artifacts generated by a workflow execution."""
    db = SessionLocal()
    try:
        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.execution_id == execution_id,
            WorkflowExecution.tenant_id == user.tenant_id,
        ).first()
        if not execution:
            raise HTTPException(status_code=404, detail="Workflow execution not found")

        return {
            "artifacts": [
                {
                    "id": a.id,
                    "step_id": a.step_id,
                    "type": a.artifact_type,
                    "title": a.title,
                    "mime_type": a.mime_type,
                    "content": a.content[:10000] if a.content else None,
                }
                for a in (execution.artifacts or [])
            ]
        }
    finally:
        db.close()


@router.get("")
async def list_workflows(
    user: UserContext = Depends(get_current_user),
):
    """List all workflow executions for the current tenant."""
    db = SessionLocal()
    try:
        executions = (
            db.query(WorkflowExecution)
            .filter(WorkflowExecution.tenant_id == user.tenant_id)
            .order_by(WorkflowExecution.created_at.desc())
            .limit(50)
            .all()
        )

        return WorkflowListResponse(
            executions=[
                WorkflowStatusResponse(
                    execution_id=e.execution_id,
                    status=ExecutionStatus(e.status),
                    progress=e.progress or 0,
                    current_step_id=e.current_step_id,
                    step_results=(e.state_json or {}).get("step_results", {}),
                    artifacts=[],
                    error_message=e.error_message,
                    goal=(e.definition.goal if e.definition else ""),
                )
                for e in executions
            ]
        )
    finally:
        db.close()


@router.delete("/{execution_id}")
async def cancel_workflow(
    execution_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Cancel a running workflow."""
    db = SessionLocal()
    try:
        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.execution_id == execution_id,
            WorkflowExecution.tenant_id == user.tenant_id,
        ).first()
        if not execution:
            raise HTTPException(status_code=404, detail="Workflow execution not found")
        # Cascade delete handles artifacts via ORM relationship
        db.delete(execution)
        db.commit()
        return {"execution_id": execution_id, "status": "deleted"}
    finally:
        db.close()
