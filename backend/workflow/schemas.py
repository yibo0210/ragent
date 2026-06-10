"""Workflow Pydantic schemas for API serialization and validation."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactType(str, Enum):
    REPORT = "report"
    EXCEL = "excel"
    CSV = "csv"
    CHART = "chart"
    DASHBOARD = "dashboard"
    PDF = "pdf"


class WorkflowStep(BaseModel):
    """A single step in a workflow DAG."""

    step_id: str = Field(..., description="Unique step identifier, e.g. 'step_1'")
    name: str = Field(..., description="Human-readable step name")
    tool: str = Field(
        ...,
        description="Agent or tool name: rag_specialist|web_searcher|data_analyst|"
        "local_graph_search|global_graph_search|direct_answer|mcp:*",
    )
    query: str = Field(..., description="Natural language task description for this step")
    dependencies: list[str] = Field(
        default_factory=list,
        description="step_ids that must complete before this step",
    )
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from dependency step_id to local variable name",
    )
    timeout: int = Field(default=60, description="Max execution seconds")


class WorkflowPlan(BaseModel):
    """Planner output: full execution plan DAG."""

    goal: str = Field(..., description="Original user goal")
    steps: list[WorkflowStep] = Field(..., description="DAG steps in execution order")
    reasoning: str = Field(default="", description="Planner reasoning chain")
    estimated_tokens: int = Field(default=0, description="Estimated token consumption")


class WorkflowArtifactRef(BaseModel):
    """Reference to a generated artifact."""

    artifact_id: int = 0
    step_id: str = ""
    artifact_type: ArtifactType = ArtifactType.REPORT
    title: str = ""
    mime_type: str = "text/markdown"
    content: str = ""
    url: str = ""


class WorkflowExecutionState(BaseModel):
    """Current state of a workflow execution, serialized for API + DB."""

    execution_id: str
    definition_id: int = 0
    tenant_id: int = 0
    user_id: int = 0
    session_id: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    plan: Optional[WorkflowPlan] = None
    current_step_id: Optional[str] = None
    step_results: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[WorkflowArtifactRef] = Field(default_factory=list)
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class WorkflowPlanRequest(BaseModel):
    """Request to generate a workflow plan."""

    goal: str = Field(..., min_length=1, description="Natural language goal")
    session_id: Optional[str] = None


class WorkflowPlanResponse(BaseModel):
    """Response containing the generated plan."""

    definition_id: int
    plan: WorkflowPlan


class WorkflowExecuteRequest(BaseModel):
    """Request to execute a workflow."""

    definition_id: int
    session_id: Optional[str] = None


class WorkflowExecuteResponse(BaseModel):
    """Response after starting workflow execution."""

    execution_id: str
    status: ExecutionStatus


class WorkflowStatusResponse(BaseModel):
    """Response for workflow status query."""

    execution_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    progress: float = 0.0
    current_step_id: Optional[str] = None
    step_results: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[WorkflowArtifactRef] = Field(default_factory=list)
    error_message: Optional[str] = None
    goal: str = ""


class WorkflowListResponse(BaseModel):
    """Response for listing workflow executions."""

    executions: list[WorkflowStatusResponse] = Field(default_factory=list)


class WorkflowTemplateBrief(BaseModel):
    """Brief info about a workflow template."""

    name: str
    description: str
    category: str
    step_count: int = 0
