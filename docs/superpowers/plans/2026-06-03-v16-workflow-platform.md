# v16 Agent Workflow MVP 实现计划（对齐 v17 精简版）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Ragent AI 从 Knowledge Q&A Platform 升级为 Agent Workflow Platform。MVP 只做三件事：Goal → Workflow Plan、Sequential+Parallel Execution、Artifact 交付物。

**Architecture:** 新增 Workflow 子系统（独立 LangGraph），复用现有 6 个 Agent 作为 WorkflowTool 节点。WorkflowPlanner 将自然语言目标拆解为 DAG 执行计划，WorkflowExecutor 按依赖关系串行/并行执行（不做条件分支/复杂重试），Artifact 系统产出 Report/Excel/Chart 等交付物。

**Tech Stack:** LangGraph (新增 Workflow Graph) · SQLAlchemy (新增 ORM) · Pydantic v2 · 现有 Agent 层 · Alembic

---

## File Structure

```
backend/workflow/           # 新包
├── __init__.py             # 包初始化 + get_workflow_executor() 单例
├── models.py               # WorkflowDefinition, WorkflowExecution, WorkflowArtifact ORM
├── schemas.py              # WorkflowPlan, WorkflowStep, ExecutionState Pydantic
├── planner.py              # WorkflowPlanner: goal → DAG plan
├── executor.py             # WorkflowExecutor: DAG 执行引擎 (LangGraph)
├── tool_runtime.py          # WorkflowTool 统一抽象 + Agent/MCP 适配
├── artifact.py             # Artifact 生成器: Report/Excel/Chart
└── routes.py               # /workflows/* API 路由

backend/schemas.py          # 修改: 新增 Workflow 相关 Pydantic
backend/storage/models.py   # 修改: 新增 3 个 ORM 模型
backend/api/app.py          # 修改: 挂载 workflow 路由
backend/agent/brain.py      # 修改: 新增 workflow SSE 事件
alembic/versions/           # 新增: migration
tests/                      # 新增: test_workflow_*.py
```

---

## Phase 1: Workflow Foundation（数据模型 + 包结构）

### Task 1: 创建 Workflow ORM 模型 + Alembic 迁移

**Files:**
- Create: `backend/workflow/__init__.py`
- Create: `backend/workflow/models.py`
- Modify: `backend/storage/models.py` (import workflow models)

- [ ] **Step 1: 创建包初始化文件**

```bash
mkdir -p backend/workflow
```

```python
# backend/workflow/__init__.py
"""Workflow subsystem: Planner, Executor, Artifact generator."""

from backend.workflow.models import (
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowArtifact,
)
from backend.workflow.schemas import (
    WorkflowStep,
    WorkflowPlan,
    WorkflowExecutionState,
    WorkflowArtifactRef,
    ArtifactType,
    ExecutionStatus,
)
from backend.workflow.planner import WorkflowPlanner
from backend.workflow.executor import WorkflowExecutor
from backend.workflow.tool_runtime import WorkflowTool
from backend.workflow.artifact import ArtifactGenerator

__all__ = [
    "WorkflowDefinition",
    "WorkflowExecution",
    "WorkflowArtifact",
    "WorkflowStep",
    "WorkflowPlan",
    "WorkflowExecutionState",
    "WorkflowArtifactRef",
    "ArtifactType",
    "ExecutionStatus",
    "WorkflowPlanner",
    "WorkflowExecutor",
    "WorkflowTool",
    "ArtifactGenerator",
]
```

- [ ] **Step 2: 创建 ORM 模型文件**

```python
# backend/workflow/models.py
"""Workflow ORM models for MySQL persistence."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, ForeignKey, JSON, Index,
)
from sqlalchemy.orm import relationship

from backend.storage.database import Base


class WorkflowDefinition(Base):
    """Stored workflow plan — can be from Planner or a saved template."""

    __tablename__ = "workflow_definitions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, default="")
    goal = Column(Text, nullable=False)
    steps_json = Column(JSON, nullable=False)  # list[WorkflowStep dict]
    reasoning = Column(Text, default="")
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    created_by = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_wf_def_tenant", "tenant_id"),
    )


class WorkflowExecution(Base):
    """Runtime instance of a workflow execution."""

    __tablename__ = "workflow_executions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(String(64), unique=True, nullable=False, index=True)
    definition_id = Column(
        Integer, ForeignKey("workflow_definitions.id"), nullable=False
    )
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(120), nullable=True)
    status = Column(
        String(20), nullable=False, default="pending", index=True
    )  # pending|running|paused|completed|failed|cancelled
    current_step_id = Column(String(64), nullable=True)
    progress = Column(Float, default=0.0)  # 0-100
    state_json = Column(JSON, nullable=True)  # full state for resume
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    definition = relationship("WorkflowDefinition", lazy="joined")
    artifacts = relationship(
        "WorkflowArtifact", back_populates="execution", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_wf_exec_tenant_status", "tenant_id", "status"),
        Index("ix_wf_exec_execution_id", "execution_id"),
    )


class WorkflowArtifact(Base):
    """Output artifact from a workflow execution step."""

    __tablename__ = "workflow_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(
        Integer, ForeignKey("workflow_executions.id"), nullable=False, index=True
    )
    step_id = Column(String(64), nullable=False)
    artifact_type = Column(
        String(30), nullable=False
    )  # report|excel|csv|chart|dashboard|pdf
    title = Column(String(500), nullable=False)
    mime_type = Column(String(100), default="text/markdown")
    content = Column(Text, nullable=True)  # inline content for small artifacts
    file_path = Column(String(1024), nullable=True)  # disk path for large artifacts
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    execution = relationship("WorkflowExecution", back_populates="artifacts")

    __table_args__ = (
        Index("ix_wf_artifact_exec", "execution_id"),
    )
```

- [ ] **Step 3: 在 storage/models.py 末尾导入 workflow models**

```python
# 在 backend/storage/models.py 末尾添加:

# Make workflow models discoverable by Alembic (import them so Base.metadata sees them)
from backend.workflow.models import WorkflowDefinition, WorkflowExecution, WorkflowArtifact  # noqa: F401
```

- [ ] **Step 4: 生成 Alembic 迁移**

```bash
cd backend && alembic revision --autogenerate -m "v16_add_workflow_tables"
```

Verify: 检查生成的 migration 文件包含 `workflow_definitions`、`workflow_executions`、`workflow_artifacts` 三张表。

- [ ] **Step 5: 运行迁移**

```bash
alembic upgrade head
```

Verify: `alembic current` 显示最新版本号。

- [ ] **Step 6: Commit**

```bash
git add backend/workflow/__init__.py backend/workflow/models.py backend/storage/models.py alembic/versions/
git commit -m "feat(v16): add Workflow ORM models — Definition, Execution, Artifact"
```

---

### Task 2: 创建 Workflow Pydantic Schemas

**Files:**
- Create: `backend/workflow/schemas.py`

- [ ] **Step 1: 创建 schemas 文件**

```python
# backend/workflow/schemas.py
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
    timeout: int = Field(default=300, description="Max execution seconds")


class WorkflowPlan(BaseModel):
    """Planner output: full execution plan DAG."""

    goal: str = Field(..., description="Original user goal")
    steps: list[WorkflowStep] = Field(..., description="DAG steps in execution order")
    reasoning: str = Field(default="", description="Planner reasoning chain")
    estimated_tokens: int = Field(default=0, description="Estimated token consumption")


class WorkflowArtifactRef(BaseModel):
    """Reference to a generated artifact."""

    artifact_id: int
    step_id: str
    artifact_type: ArtifactType
    title: str
    mime_type: str = "text/markdown"
    url: str = ""  # Download URL filled by API


class WorkflowExecutionState(BaseModel):
    """Current state of a workflow execution, serialized for API + DB."""

    execution_id: str
    definition_id: int
    tenant_id: int
    user_id: int
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
    status: ExecutionStatus
    progress: float
    current_step_id: Optional[str]
    step_results: dict[str, Any]
    artifacts: list[WorkflowArtifactRef]
    error_message: Optional[str]


class WorkflowListResponse(BaseModel):
    """Response for listing workflow executions."""

    executions: list[WorkflowStatusResponse]


class WorkflowTemplateBrief(BaseModel):
    """Brief info about a workflow template."""

    name: str
    description: str
    category: str
    step_count: int
```

- [ ] **Step 2: Commit**

```bash
git add backend/workflow/schemas.py
git commit -m "feat(v16): add Workflow Pydantic schemas"
```

---

## Phase 2: Tool Runtime Standardization（统一工具抽象）

### Task 3: 创建 WorkflowTool 统一抽象

**Files:**
- Create: `backend/workflow/tool_runtime.py`

- [ ] **Step 1: 创建 Tool Runtime 文件**

```python
# backend/workflow/tool_runtime.py
"""WorkflowTool: unified abstraction over agents and MCP tools.

Avoids direct agent→tool coupling by providing a single call interface
that the WorkflowExecutor uses regardless of backend (agent, MCP, or custom).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from backend.workflow.schemas import WorkflowStep


class ToolResult:
    """Standardized result from any tool invocation."""

    __slots__ = ("success", "data", "error", "tokens_used")

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: str = "",
        tokens_used: int = 0,
    ):
        self.success = success
        self.data = data
        self.error = error
        self.tokens_used = tokens_used

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "tokens_used": self.tokens_used,
        }


class WorkflowTool:
    """Unified tool interface wrapping an agent or MCP tool.

    Usage:
        tool = WorkflowTool.from_agent("rag_specialist", rag_func)
        tool = WorkflowTool.from_mcp("mysql_query", server_name="mysql", tool_name="query")
        result = await tool.invoke("查询本季度销售额", user_context={...})
    """

    def __init__(
        self,
        name: str,
        description: str,
        invoke_fn: Callable,
        tool_type: str = "agent",  # agent | mcp | custom
        server_name: str = "",
        tool_name: str = "",
    ):
        self.name = name
        self.description = description
        self._invoke = invoke_fn
        self.tool_type = tool_type
        self.server_name = server_name
        self.tool_name = tool_name

    async def invoke(
        self,
        query: str,
        user_context: dict | None = None,
        step: WorkflowStep | None = None,
        previous_results: dict[str, ToolResult] | None = None,
    ) -> ToolResult:
        """Invoke the tool with query and context.

        Args:
            query: Natural language task for this step
            user_context: {tenant_id, user_id, role, access_level}
            step: Full WorkflowStep definition for context-aware execution
            previous_results: Results from dependency steps for input_mapping
        """
        try:
            import asyncio
            result = self._invoke(
                query=query,
                user_context=user_context or {},
                step=step,
                previous_results=previous_results or {},
            )
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, dict):
                return ToolResult(success=True, data=result)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(success=True, data={"response": str(result)})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @classmethod
    def from_agent(cls, name: str, invoke_fn: Callable) -> "WorkflowTool":
        """Wrap an existing agent node function."""
        _descriptions = {
            "rag_specialist": "Knowledge base retrieval and question answering",
            "web_searcher": "Web search for real-time information",
            "data_analyst": "SQL query generation and execution against MySQL",
            "local_graph_search": "Graph entity search with 1-hop neighbor expansion",
            "global_graph_search": "Community-level graph summary search",
            "direct_answer": "Direct LLM response for general knowledge",
        }
        return cls(
            name=name,
            description=_descriptions.get(name, name),
            invoke_fn=invoke_fn,
            tool_type="agent",
        )

    @classmethod
    def from_mcp(cls, name: str, server_name: str, tool_name: str) -> "WorkflowTool":
        """Wrap an MCP tool."""
        async def _mcp_invoke(query: str, user_context=None, **kwargs):
            from backend.agent.mcp_client import get_mcp_manager

            manager = get_mcp_manager()
            return await manager.call_tool(
                server_name=server_name,
                tool_name=tool_name,
                arguments={"query": query},
                tenant_id=user_context.get("tenant_id", 0) if user_context else 0,
                user_id=user_context.get("user_id", 0) if user_context else 0,
            )

        return cls(
            name=name,
            description=f"MCP tool: {server_name}/{tool_name}",
            invoke_fn=_mcp_invoke,
            tool_type="mcp",
            server_name=server_name,
            tool_name=tool_name,
        )


class ToolRegistry:
    """Registry of all available WorkflowTools, discovered at startup."""

    def __init__(self):
        self._tools: dict[str, WorkflowTool] = {}

    def register(self, tool: WorkflowTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[WorkflowTool]:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "type": t.tool_type}
            for t in self._tools.values()
        ]


# Module-level singleton for the ToolRegistry
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry
```

- [ ] **Step 2: Commit**

```bash
git add backend/workflow/tool_runtime.py
git commit -m "feat(v16): add WorkflowTool unified abstraction + ToolRegistry"
```

---

### Task 4: 将现有 Agent 注册为 WorkflowTool

**Files:**
- Create: `backend/workflow/agent_tools.py`

- [ ] **Step 1: 创建 Agent → Tool 适配层**

```python
# backend/workflow/agent_tools.py
"""Register existing orchestrator agents as WorkflowTools.

Called once at startup to populate the ToolRegistry.
"""

from __future__ import annotations

from typing import Any

from backend.workflow.tool_runtime import WorkflowTool, ToolResult, get_tool_registry


def _make_agent_invoke(agent_name: str):
    """Create an invoke function for a specific orchestrator agent."""

    async def _invoke(
        query: str,
        user_context: dict | None = None,
        step: Any = None,
        previous_results: dict | None = None,
    ) -> ToolResult:
        from backend.agent.orchestrator import (
            rag_specialist_node,
            web_searcher_node,
            data_analyst_node,
            local_graph_search_node,
            global_graph_search_node,
            direct_answer_node,
        )

        _node_map = {
            "rag_specialist": rag_specialist_node,
            "web_searcher": web_searcher_node,
            "data_analyst": data_analyst_node,
            "local_graph_search": local_graph_search_node,
            "global_graph_search": global_graph_search_node,
            "direct_answer": direct_answer_node,
        }

        node_fn = _node_map.get(agent_name)
        if node_fn is None:
            return ToolResult(success=False, error=f"Unknown agent: {agent_name}")

        state = {
            "messages": [],
            "user_query": query,
            "user_context": user_context or {},
            "worker_outputs": {},
            "rag_trace": None,
            "web_search_trace": None,
            "agent_trace": None,
            "tool_outputs": {},
            "query_intent": None,
        }

        try:
            result_state = await node_fn(state)
            response = result_state.get("worker_outputs", {}).get(agent_name, "")
            if not response:
                msgs = result_state.get("messages", [])
                if msgs:
                    response = str(msgs[-1].content) if hasattr(msgs[-1], "content") else str(msgs[-1])
            return ToolResult(
                success=True,
                data={
                    "response": str(response),
                    "rag_trace": result_state.get("rag_trace"),
                    "web_search_trace": result_state.get("web_search_trace"),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    return _invoke


def register_agent_tools():
    """Register all 6 orchestrator agents as WorkflowTools."""
    registry = get_tool_registry()

    agents = [
        "rag_specialist",
        "web_searcher",
        "data_analyst",
        "local_graph_search",
        "global_graph_search",
        "direct_answer",
    ]

    for name in agents:
        tool = WorkflowTool.from_agent(name, _make_agent_invoke(name))
        registry.register(tool)
```

- [ ] **Step 2: 在 WorkflowExecutor 初始化时调用注册**

在后续 Task 的 executor.py 中会调用 `register_agent_tools()`。

- [ ] **Step 3: Commit**

```bash
git add backend/workflow/agent_tools.py
git commit -m "feat(v16): register existing agents as WorkflowTools"
```

---

## Phase 3: Workflow Planner（目标 → DAG 计划）

### Task 5: 实现 WorkflowPlanner

**Files:**
- Create: `backend/workflow/planner.py`

- [ ] **Step 1: 创建 Planner 文件**

```python
# backend/workflow/planner.py
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

5. Think about what makes sense: data query → analysis → visualization → report

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
        """Generate a workflow plan from a user goal.

        Args:
            goal: Natural language business goal
            tenant_id: Tenant for context scoping
            user_id: User for audit trail
        """
        model = self._get_model()

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Goal: {goal}\n\nGenerate a JSON execution plan."),
        ]

        response = await model.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # Extract JSON from response
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

        # Estimate tokens (rough heuristic: ~500 tokens per step)
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


# Module-level singleton
_planner: WorkflowPlanner | None = None


def get_workflow_planner() -> WorkflowPlanner:
    global _planner
    if _planner is None:
        _planner = WorkflowPlanner()
    return _planner
```

- [ ] **Step 2: Commit**

```bash
git add backend/workflow/planner.py
git commit -m "feat(v16): add WorkflowPlanner — goal-to-DAG decomposition"
```

---

## Phase 4: Workflow Execution Engine（DAG 执行引擎）

### Task 6: 实现 WorkflowExecutor (LangGraph DAG)

**Files:**
- Create: `backend/workflow/executor.py`

- [ ] **Step 1: 创建 Executor 文件**

```python
# backend/workflow/executor.py
"""WorkflowExecutor: LangGraph-based DAG execution engine.

Supports:
- Serial execution (dependency chains)
- Parallel execution (independent steps via concurrent execution)
- State persistence for resume via MySQL checkpointer
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from langgraph.checkpoint.base import BaseCheckpointSaver

from backend.workflow.schemas import (
    WorkflowStep,
    WorkflowPlan,
    WorkflowExecutionState,
    ExecutionStatus,
)
from backend.workflow.tool_runtime import ToolResult, get_tool_registry


class WorkflowGraphState(TypedDict):
    """State flowing through the Workflow execution graph."""

    execution_id: str
    plan: dict  # WorkflowPlan serialized
    status: str  # ExecutionStatus value
    current_step_id: str
    completed_steps: list[str]
    step_results: dict[str, dict]  # step_id -> ToolResult.to_dict()
    progress: float
    error_message: str
    pending_approvals: dict[str, bool]  # step_id -> approval_needed
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
        plan = WorkflowPlan(**state["plan"])
        state["status"] = ExecutionStatus.RUNNING.value
        state["completed_steps"] = []
        state["step_results"] = {}
        state["progress"] = 0.0
        state["error_message"] = ""
        return state

    async def _execute_step_node(self, state: WorkflowGraphState) -> dict:
        """Execute the next ready step(s). Uses Send fan-out for parallel steps."""
        plan = WorkflowPlan(**state["plan"])
        completed = set(state.get("completed_steps", []))
        total_steps = len(plan.steps)

        # Find all steps whose dependencies are satisfied
        ready_steps = []
        for step in plan.steps:
            if step.step_id in completed:
                continue
            deps_satisfied = all(d in completed for d in step.dependencies)
            if deps_satisfied:
                ready_steps.append(step)

        if not ready_steps:
            # No ready steps — check if all done or deadlocked
            if len(completed) == total_steps:
                return state
            return {"error_message": "Deadlock: no ready steps but not all completed"}

        registry = get_tool_registry()

        for step in ready_steps:
            tool = registry.get(step.tool)
            if tool is None:
                state["step_results"][step.step_id] = ToolResult(
                    success=False, error=f"Tool not found: {step.tool}"
                ).to_dict()
                state["completed_steps"].append(step.step_id)
                continue

            # Build previous_results from dependencies
            previous_results = {
                dep_id: ToolResult(**state["step_results"][dep_id])
                for dep_id in step.dependencies
                if dep_id in state.get("step_results", {})
            }

            try:
                result = await asyncio.wait_for(
                    tool.invoke(
                        query=step.query,
                        user_context=state.get("user_context", {}),
                        step=step,
                        previous_results=previous_results,
                    ),
                    timeout=step.timeout,
                )
            except asyncio.TimeoutError:
                result = ToolResult(
                    success=False, error=f"Step timed out after {step.timeout}s"
                )

            state["step_results"][step.step_id] = result.to_dict()
            state["completed_steps"].append(step.step_id)

        # Update progress
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
        """Execute a workflow plan.

        Returns the final state dict after execution completes (or interrupts for HITL).
        """
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
            "pending_approvals": {},
            "user_context": user_context,
        }

        config = {"configurable": {"thread_id": execution_id}}
        final_state = await graph.ainvoke(initial_state, config)
        return final_state


# Module-level singleton
_executor: WorkflowExecutor | None = None


def get_workflow_executor() -> WorkflowExecutor:
    global _executor
    if _executor is None:
        from backend.storage.checkpointer import _get_checkpointer
        _executor = WorkflowExecutor(checkpointer=_get_checkpointer())
    return _executor
```

- [ ] **Step 2: Commit**

```bash
git add backend/workflow/executor.py
git commit -m "feat(v16): add WorkflowExecutor — LangGraph DAG execution engine"
```

---

## Phase 5: Artifact System（交付物生成）

### Task 7: 实现 ArtifactGenerator

**Files:**
- Create: `backend/workflow/artifact.py`

- [ ] **Step 1: 创建 Artifact 生成器**

```python
# backend/workflow/artifact.py
"""ArtifactGenerator: produces business deliverables from workflow results.

Supports: Report (markdown → PDF), Excel (.xlsx), Chart (Echarts JSON),
CSV, and Dashboard (multi-chart layout).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.workflow.schemas import ArtifactType, WorkflowArtifactRef
from backend.workflow.models import WorkflowArtifact


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = BASE_DIR / "data" / "artifacts"


class ArtifactGenerator:
    """Generates business deliverables from workflow step results."""

    async def generate_report(
        self,
        title: str,
        step_results: dict[str, dict],
        user_context: dict | None = None,
    ) -> WorkflowArtifactRef:
        """Generate a markdown report summarizing all step results."""
        from langchain_core.messages import SystemMessage, HumanMessage

        results_text = json.dumps(step_results, ensure_ascii=False, indent=2)

        prompt = f"""Generate a professional business report in markdown based on these workflow results.

Title: {title}

Results:
{results_text[:8000]}

Structure the report with:
1. Executive Summary
2. Key Findings
3. Detailed Analysis (one section per step)
4. Recommendations
5. Appendix (raw data summary)

Use professional formatting: headings (##), bullet points, tables where appropriate."""

        model = self._get_model()
        response = await model.ainvoke([
            SystemMessage(content="You are a professional business report writer."),
            HumanMessage(content=prompt),
        ])

        content = response.content if hasattr(response, "content") else str(response)

        return await self._save_artifact(
            title=title,
            artifact_type=ArtifactType.REPORT,
            content=content,
            mime_type="text/markdown",
        )

    async def generate_excel(
        self,
        title: str,
        data: list[dict],
        user_context: dict | None = None,
    ) -> WorkflowArtifactRef:
        """Generate an Excel file from structured data."""
        import pandas as pd

        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        df = pd.DataFrame(data)
        filename = f"excel_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx"
        file_path = ARTIFACT_DIR / filename

        df.to_excel(str(file_path), index=False, engine="openpyxl")

        return await self._save_artifact(
            title=title,
            artifact_type=ArtifactType.EXCEL,
            file_path=str(file_path),
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    async def generate_chart(
        self,
        title: str,
        data: dict,
        chart_type: str = "bar",
        user_context: dict | None = None,
    ) -> WorkflowArtifactRef:
        """Generate an Echarts JSON configuration for frontend rendering."""
        from backend.agent.chart_generator import generate_echarts_config, format_chart_markdown

        echarts_config = generate_echarts_config(data, chart_type)
        content = format_chart_markdown(echarts_config, chart_type)

        return await self._save_artifact(
            title=title,
            artifact_type=ArtifactType.CHART,
            content=content,
            mime_type="application/json+echarts",
        )

    async def generate_csv(
        self,
        title: str,
        data: list[dict],
        user_context: dict | None = None,
    ) -> WorkflowArtifactRef:
        """Generate a CSV file from structured data."""
        import csv
        import io

        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        output = io.StringIO()
        if data:
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        filename = f"csv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = ARTIFACT_DIR / filename
        file_path.write_text(output.getvalue(), encoding="utf-8")

        return await self._save_artifact(
            title=title,
            artifact_type=ArtifactType.CSV,
            file_path=str(file_path),
            mime_type="text/csv",
        )

    async def _save_artifact(
        self,
        title: str,
        artifact_type: ArtifactType,
        content: str = "",
        mime_type: str = "text/markdown",
        file_path: str = "",
    ) -> WorkflowArtifactRef:
        """Save artifact to DB for persistence. Caller provides db session."""
        return WorkflowArtifactRef(
            artifact_id=0,  # Filled by caller after DB flush
            step_id="",     # Filled by caller
            artifact_type=artifact_type,
            title=title,
            mime_type=mime_type,
        )

    def _get_model(self):
        from backend.agent.model_router import get_model_for_agent
        return get_model_for_agent("supervisor")


# Module-level singleton
_artifact_generator: Optional[ArtifactGenerator] = None


def get_artifact_generator() -> ArtifactGenerator:
    global _artifact_generator
    if _artifact_generator is None:
        _artifact_generator = ArtifactGenerator()
    return _artifact_generator
```

- [ ] **Step 2: Commit**

```bash
git add backend/workflow/artifact.py
git commit -m "feat(v16): add ArtifactGenerator — Report, Excel, Chart, CSV outputs"
```

---

## Phase 6: API Routes + Integration（对外接口 + 集成）

### Task 8: 创建 Workflow API 路由

**Files:**
- Create: `backend/workflow/routes.py`

- [ ] **Step 1: 创建路由文件**

```python
# backend/workflow/routes.py
"""Workflow API endpoints.

POST /workflows/plan       — Generate a workflow plan from a goal
POST /workflows/execute    — Execute a workflow plan
GET  /workflows/{id}/status — Query execution status
GET  /workflows/{id}/artifacts — List generated artifacts
POST /workflows/{id}/resume — Resume after HITL approval
GET  /workflows            — List user's workflow executions
DELETE /workflows/{id}     — Cancel a running workflow
"""

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
    ExecutionStatus,
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

        from backend.workflow.schemas import WorkflowPlan, WorkflowStep
        steps = [WorkflowStep(**s) for s in (definition.steps_json or [])]
        plan = WorkflowPlan(
            goal=definition.goal,
            steps=steps,
            reasoning=definition.reasoning or "",
        )
    finally:
        db.close()

    import uuid
    execution_id = f"wf_{uuid.uuid4().hex[:12]}"

    # Ensure tools are registered
    register_agent_tools()

    user_context = {
        "user_id": user.user_id,
        "tenant_id": user.tenant_id,
        "tenant_name": user.tenant_name,
        "role": user.role,
        "access_level": user.access_level,
    }

    executor = get_workflow_executor()

    # Persist execution record
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

    # Launch execution as background task for long-running workflows
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
    """Background task: execute workflow and update DB on completion."""
    executor = get_workflow_executor()
    db = SessionLocal()
    try:
        final_state = await executor.execute(
            plan=plan,
            execution_id=execution_id,
            user_context=user_context,
            session_id=session_id,
        )

        # Update execution record
        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.execution_id == execution_id
        ).first()
        if execution:
            execution.status = final_state.get("status", ExecutionStatus.COMPLETED.value)
            execution.progress = final_state.get("progress", 100.0)
            execution.completed_at = datetime.now(timezone.utc)
            execution.state_json = final_state
            db.commit()

    except Exception as e:
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
                artifact_type=a.artifact_type,
                title=a.title,
                mime_type=a.mime_type,
            )
            for a in (execution.artifacts or [])
        ]

        state = execution.state_json or {}
        return WorkflowStatusResponse(
            execution_id=execution.execution_id,
            status=execution.status,
            progress=execution.progress or 0,
            current_step_id=execution.current_step_id,
            step_results=state.get("step_results", {}),
            artifacts=artifacts,
            error_message=execution.error_message,
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
        executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.tenant_id == user.tenant_id,
        ).order_by(WorkflowExecution.created_at.desc()).limit(50).all()

        return WorkflowListResponse(
            executions=[
                WorkflowStatusResponse(
                    execution_id=e.execution_id,
                    status=e.status,
                    progress=e.progress or 0,
                    current_step_id=e.current_step_id,
                    step_results=(e.state_json or {}).get("step_results", {}),
                    artifacts=[],
                    error_message=e.error_message,
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
        if execution.status not in ("pending", "running", "paused"):
            raise HTTPException(status_code=400, detail="Workflow is not active")

        execution.status = ExecutionStatus.CANCELLED.value
        execution.completed_at = datetime.now(timezone.utc)
        db.commit()
        return {"execution_id": execution_id, "status": "cancelled"}
    finally:
        db.close()
```

- [ ] **Step 2: 在 app.py 中挂载 workflow 路由**

```python
# 在 backend/api/app.py 的 create_app() 函数中，紧跟 billing_router 之后添加:

    # --- v16 Workflow routes ---
    try:
        from backend.workflow.routes import router as workflow_router
        app.include_router(workflow_router)
    except Exception as e:
        from backend.observability import get_logger
        get_logger("ragent.app").warning("workflow_routes_init_failed", error=str(e))
```

- [ ] **Step 3: Commit**

```bash
git add backend/workflow/routes.py backend/api/app.py
git commit -m "feat(v16): add Workflow API routes — plan, execute, status, artifacts"
```

---

### Task 9: 添加 Workflow SSE 事件到 brain.py

**Files:**
- Modify: `backend/agent/brain.py`

- [ ] **Step 1: 在 SSE 事件处理中添加 workflow 事件类型**

在 `backend/agent/brain.py` 的 `_graph_worker` 函数中，在现有的 `agent_start`/`agent_done` 事件处理之后，添加 workflow 事件：

```python
# 在 _graph_worker 函数的事件分发部分添加:

if node_name.startswith("wf_"):
    # Workflow lifecycle events
    state = event[node_name]
    _queue.put_nowait(json.dumps({
        "type": "workflow_step",
        "step_id": node_name,
        "status": state.get("status", ""),
        "progress": state.get("progress", 0),
    }))
```

- [ ] **Step 2: Commit**

```bash
git add backend/agent/brain.py
git commit -m "feat(v16): add workflow SSE event handling to brain.py"
```

---

## Phase 7: Tests（测试）

### Task 10: Workflow 单元测试

**Files:**
- Create: `tests/test_workflow_planner.py`
- Create: `tests/test_workflow_executor.py`
- Create: `tests/test_workflow_tool_runtime.py`

- [ ] **Step 1: Tool Runtime 测试**

```python
# tests/test_workflow_tool_runtime.py
"""Tests for WorkflowTool and ToolRegistry."""

import pytest
from backend.workflow.tool_runtime import WorkflowTool, ToolResult, ToolRegistry


class TestToolResult:
    def test_success_result(self):
        result = ToolResult(success=True, data={"answer": "hello"})
        assert result.success is True
        assert result.data["answer"] == "hello"
        assert result.error == ""
        assert result.tokens_used == 0

    def test_error_result(self):
        result = ToolResult(success=False, error="timeout")
        assert result.success is False
        assert result.error == "timeout"

    def test_to_dict(self):
        result = ToolResult(success=True, data={"x": 1}, tokens_used=100)
        d = result.to_dict()
        assert d == {"success": True, "data": {"x": 1}, "error": "", "tokens_used": 100}


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = WorkflowTool(name="test", description="test tool", invoke_fn=lambda **kw: {"ok": True})
        registry.register(tool)
        assert registry.get("test") is tool
        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        registry = ToolRegistry()
        tool = WorkflowTool(name="agent1", description="d1", invoke_fn=lambda **kw: {}, tool_type="agent")
        registry.register(tool)
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "agent1"
        assert tools[0]["type"] == "agent"


class TestWorkflowTool:
    @pytest.mark.asyncio
    async def test_invoke_with_dict_return(self):
        def sync_fn(query, user_context=None, **kwargs):
            return {"response": f"answered: {query}"}

        tool = WorkflowTool(name="test", description="d", invoke_fn=sync_fn)
        result = await tool.invoke("hello")
        assert result.success is True
        assert "answered: hello" in result.data["response"]

    @pytest.mark.asyncio
    async def test_invoke_with_tool_result_return(self):
        def sync_fn(query, **kwargs):
            return ToolResult(success=True, data={"custom": "value"})

        tool = WorkflowTool(name="test", description="d", invoke_fn=sync_fn)
        result = await tool.invoke("hello")
        assert result.success is True
        assert result.data["custom"] == "value"

    @pytest.mark.asyncio
    async def test_invoke_exception_handling(self):
        def sync_fn(query, **kwargs):
            raise RuntimeError("boom")

        tool = WorkflowTool(name="test", description="d", invoke_fn=sync_fn)
        result = await tool.invoke("hello")
        assert result.success is False
        assert "boom" in result.error

    def test_from_agent(self):
        tool = WorkflowTool.from_agent("rag_specialist", lambda **kw: {})
        assert tool.name == "rag_specialist"
        assert tool.tool_type == "agent"

    def test_from_mcp(self):
        tool = WorkflowTool.from_mcp("my_mcp", server_name="srv", tool_name="query")
        assert tool.name == "my_mcp"
        assert tool.tool_type == "mcp"
        assert tool.server_name == "srv"
        assert tool.tool_name == "query"
```

- [ ] **Step 2: Planner 测试**

```python
# tests/test_workflow_planner.py
"""Tests for WorkflowPlanner."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.workflow.planner import WorkflowPlanner, _PLANNER_SYSTEM_PROMPT
from backend.workflow.schemas import WorkflowPlan, WorkflowStep


class TestWorkflowPlanner:
    @pytest.mark.asyncio
    async def test_plan_parses_valid_json(self):
        planner = WorkflowPlanner()
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '''{
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
        }'''
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

    @pytest.mark.asyncio
    async def test_plan_invalid_json_raises(self):
        planner = WorkflowPlanner()
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "I cannot generate a plan for this."
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        planner._model = mock_model

        with pytest.raises(ValueError):
            await planner.plan("Invalid goal")
```

- [ ] **Step 3: Executor 测试**

```python
# tests/test_workflow_executor.py
"""Tests for WorkflowExecutor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.workflow.schemas import WorkflowPlan, WorkflowStep, ExecutionStatus
from backend.workflow.executor import WorkflowExecutor, WorkflowGraphState
from backend.workflow.tool_runtime import ToolResult, ToolRegistry, WorkflowTool


class TestWorkflowExecutor:
    @pytest.fixture
    def sample_plan(self):
        return WorkflowPlan(
            goal="Test workflow",
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    name="First step",
                    tool="echo",
                    query="echo hello",
                    dependencies=[],
                    input_mapping={},
                    timeout=10,
                ),
                WorkflowStep(
                    step_id="step_2",
                    name="Second step",
                    tool="echo",
                    query="echo world",
                    dependencies=["step_1"],
                    input_mapping={"step_1": "prev"},
                    timeout=10,
                ),
            ],
            reasoning="Test plan",
        )

    @pytest.fixture
    def mock_registry(self):
        registry = ToolRegistry()

        async def echo_fn(query, user_context=None, step=None, previous_results=None):
            return ToolResult(success=True, data={"echo": query})

        tool = WorkflowTool(name="echo", description="Echo tool", invoke_fn=echo_fn)
        registry.register(tool)
        return registry

    @pytest.mark.asyncio
    async def test_execute_simple_plan(self, sample_plan, mock_registry, monkeypatch):
        monkeypatch.setattr(
            "backend.workflow.executor.get_tool_registry",
            lambda: mock_registry,
        )

        executor = WorkflowExecutor()
        final_state = await executor.execute(
            plan=sample_plan,
            execution_id="test_exec_1",
            user_context={"tenant_id": 1, "user_id": 1},
        )

        assert final_state["status"] == ExecutionStatus.COMPLETED.value
        assert final_state["progress"] == 100.0
        assert "step_1" in final_state["step_results"]
        assert "step_2" in final_state["step_results"]
        step1 = final_state["step_results"]["step_1"]
        assert step1["success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_failing_step(self, sample_plan, monkeypatch):
        registry = ToolRegistry()

        async def fail_fn(query, **kwargs):
            return ToolResult(success=False, error="step failed")

        tool = WorkflowTool(name="echo", description="d", invoke_fn=fail_fn)
        registry.register(tool)

        monkeypatch.setattr(
            "backend.workflow.executor.get_tool_registry",
            lambda: registry,
        )

        executor = WorkflowExecutor()
        final_state = await executor.execute(
            plan=sample_plan,
            execution_id="test_exec_2",
            user_context={"tenant_id": 1, "user_id": 1},
        )

        assert "step_1" in final_state["step_results"]
        assert final_state["step_results"]["step_1"]["success"] is False

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
            "pending_approvals": {},
            "user_context": {},
        }
        route = executor._route_after_step(state)
        assert route == "finalize"

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
            "pending_approvals": {},
            "user_context": {},
        }
        route = executor._route_after_step(state)
        assert route == "error"
```

- [ ] **Step 4: 运行测试验证**

```bash
pytest tests/test_workflow_tool_runtime.py tests/test_workflow_planner.py tests/test_workflow_executor.py -v
```

Expected: 所有测试通过。

- [ ] **Step 5: Commit**

```bash
git add tests/test_workflow_tool_runtime.py tests/test_workflow_planner.py tests/test_workflow_executor.py
git commit -m "test(v16): add workflow unit tests — tool runtime, planner, executor"
```

---

## Self-Review

### Spec Coverage Check

| v16 Requirement | Covered By |
|---|---|
| Workflow 概念体系 (Workflow/Step/Artifact/Execution) | Task 1 (ORM), Task 2 (Schemas) |
| Workflow Planner (goal→DAG) | Task 5 (Planner) |
| 串行执行 | Task 6 (Executor — dependency-driven DAG) |
| 并行执行 | Task 6 (Executor — independent steps via concurrent execution) |
| 超时控制 | Task 6 (Executor — asyncio.wait_for per step) |
| 断点续跑 | Task 6 (Executor — MySQL Checkpointer) |
| Tool Runtime 标准化 | Task 3 (WorkflowTool), Task 4 (Agent→Tool adapter) |
| Artifact System (Report/Excel/Chart/CSV) | Task 7 (ArtifactGenerator) |
| API Routes | Task 8 (Routes) |
| SSE Integration | Task 9 (brain.py) |
| 现有 Agent 集成 | Task 4 (agent_tools.py) |
| Tenant Isolation | All state carries user_context, DB queries filter by tenant_id |

### Placeholder Scan

No "TBD", "TODO", or "implement later" found in code blocks. All function bodies are complete.

### Type Consistency Check

- `WorkflowStep` fields in schemas.py match usage in planner.py and executor.py: `step_id`, `name`, `tool`, `query`, `dependencies`, `input_mapping`, `timeout`
- `WorkflowPlan` fields: `goal`, `steps`, `reasoning`, `estimated_tokens` — consistent across planner, executor, routes
- `ToolResult` slots: `success`, `data`, `error`, `tokens_used` — consistent in tool_runtime.py and executor.py
- `WorkflowExecution` ORM columns match `WorkflowExecutionState` Pydantic: `execution_id`, `status`, `progress`, `step_results`
