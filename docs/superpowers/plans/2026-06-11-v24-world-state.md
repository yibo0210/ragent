# v24 World State Architecture 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 引入 World State 层，使 Agent 从"只理解对话"升级为"理解真实工作环境"。维护 current_goal、active_tasks、resources、artifacts、open_contexts、environment 六大维度状态，Agent 在执行任何操作前都能感知当前全局状态，实现跨任务上下文延续。

**Architecture:** 新增 `backend/world/` 包，包含 4 个模块。StateManager 作为 World State 的单一数据源（Redis 热数据 + MySQL 持久化），维护 `WorldState` 六维模型。ArtifactRegistry 追踪所有 Artifact（Research Report/Excel/Chart/PDF/PPTX）的位置、版本和状态。ResourceRegistry 管理 Agent 可用的计算资源（MCP 连接、API Key、数据库连接池）的可用性。ContextManager 支持 Agent 在多任务间切换时保存/恢复执行上下文（open_contexts），实现暂停→恢复的上下文连续性。World State 通过 SupervisorState 注入所有 Agent 节点，替换目前靠 state 字段隐式传递的方式。

**Tech Stack:** Redis 7.0（热状态缓存）· MySQL 8.0（World State 持久化）· Pydantic v2 · 复用 SupervisorState · 复用 v14 tenant isolation

---

## File Structure

```
backend/world/                             # 新增包
├── __init__.py                            # 包入口 + 单例工厂
├── schemas.py                             # WorldState, ArtifactRef, ResourceRef, ContextSnapshot
├── state_manager.py                       # World State CRUD (Redis + MySQL)
├── artifact_registry.py                   # Artifact 注册/发现/版本追踪
├── resource_registry.py                   # 资源可用性管理
├── context_manager.py                     # 上下文保存/恢复/切换

backend/agent/orchestrator.py              # 修改: supervisor_node 读取并更新 World State
backend/research/executor.py               # 修改: 研究任务注册/更新 Artifact
backend/agent/brain.py                     # 修改: SSE 推送 world_state 事件

tests/test_world_state.py                  # 新增: World State 测试
```

---

## Phase 1: Schemas + State Manager

### Task 1: World State Schemas

**Files:**
- Create: `backend/world/__init__.py`
- Create: `backend/world/schemas.py`

```python
# backend/world/schemas.py
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ArtifactType(str, Enum):
    REPORT = "report"
    EXCEL = "excel"
    CHART = "chart"
    PDF = "pdf"
    PPTX = "pptx"
    CSV = "csv"
    OTHER = "other"


class ArtifactStatus(str, Enum):
    DRAFT = "draft"
    COMPLETE = "complete"
    ARCHIVED = "archived"


class ResourceType(str, Enum):
    MCP_SERVER = "mcp_server"
    DATABASE = "database"
    API_KEY = "api_key"
    LLM_MODEL = "llm_model"


class ArtifactRef(BaseModel):
    artifact_id: str = ""
    artifact_type: ArtifactType = ArtifactType.REPORT
    title: str = ""
    version: int = 1
    status: ArtifactStatus = ArtifactStatus.DRAFT
    file_path: str = ""
    created_at: str = ""
    updated_at: str = ""
    dependencies: list[str] = Field(default_factory=list)  # depends on artifact_ids


class ResourceRef(BaseModel):
    resource_id: str = ""
    resource_type: ResourceType = ResourceType.MCP_SERVER
    name: str = ""
    is_available: bool = True
    last_checked: str = ""
    metadata: dict = Field(default_factory=dict)


class ContextSnapshot(BaseModel):
    """Snapshot of an agent's execution context for pause/resume."""
    context_id: str = ""
    goal: str = ""
    completed_steps: list[str] = Field(default_factory=list)
    pending_steps: list[str] = Field(default_factory=list)
    intermediate_results: dict = Field(default_factory=dict)
    agent_state: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WorldState(BaseModel):
    """Global world state visible to all agents."""
    state_id: str = ""
    tenant_id: int = 0
    user_id: int = 0

    current_goal: str = ""                  # 当前最高优先级目标
    active_tasks: list[str] = Field(default_factory=list)     # 进行中的任务
    queued_tasks: list[str] = Field(default_factory=list)     # 等待执行的任务

    artifacts: list[ArtifactRef] = Field(default_factory=list)
    resources: list[ResourceRef] = Field(default_factory=list)
    open_contexts: list[ContextSnapshot] = Field(default_factory=list)

    environment: dict = Field(default_factory=dict)  # tenant tier, timezone, locale

    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
```

### Task 2: State Manager

**Files:**
- Create: `backend/world/state_manager.py`

```python
# backend/world/state_manager.py
"""StateManager: single source of truth for World State (Redis hot + MySQL cold)."""

import json
from backend.storage.cache import get_redis
from backend.storage.database import get_db

_WORLD_STATE_KEY = "world_state:{tenant_id}:{user_id}"
_WORLD_STATE_TTL = 3600  # 1 hour hot cache

class StateManager:
    def get_state(self, tenant_id: int, user_id: int) -> WorldState:
        """Get current world state (Redis cache → MySQL → new empty)."""
        ...

    def update_goal(self, tenant_id: int, user_id: int, goal: str):
        """Update current goal and persist."""
        ...

    def add_task(self, tenant_id: int, user_id: int, task_id: str):
        """Register a new active task."""
        ...

    def complete_task(self, tenant_id: int, user_id: int, task_id: str):
        """Move task from active to completed."""
        ...

    def get_status_summary(self, tenant_id: int, user_id: int) -> str:
        """Format world state as a human-readable summary for LLM injection."""
        state = self.get_state(tenant_id, user_id)
        lines = ["## 当前环境状态"]
        lines.append(f"目标: {state.current_goal or '无'}")
        lines.append(f"进行中: {len(state.active_tasks)} 个任务")
        lines.append(f"制品: {len(state.artifacts)} 个")
        lines.append(f"可用资源: {sum(1 for r in state.resources if r.is_available)}/{len(state.resources)}")
        return "\n".join(lines)
```

---

## Phase 2: Registries + Context Manager

### Task 3: Artifact Registry + Resource Registry + Context Manager

**Files:**
- Create: `backend/world/artifact_registry.py`
- Create: `backend/world/resource_registry.py`
- Create: `backend/world/context_manager.py`

```python
# backend/world/artifact_registry.py
"""ArtifactRegistry: tracks all agent-produced artifacts across workflows."""

class ArtifactRegistry:
    def register(self, tenant_id: int, user_id: int, artifact: ArtifactRef):
        """Register a new or updated artifact."""
        ...

    def get_by_id(self, artifact_id: str) -> ArtifactRef | None:
        ...

    def list_by_status(self, tenant_id: int, user_id: int, status: ArtifactStatus) -> list[ArtifactRef]:
        ...

    def get_dependency_graph(self, artifact_id: str) -> dict:
        """Return {artifact_id: [depends_on...]} for visualization."""
        ...


# backend/world/resource_registry.py
"""ResourceRegistry: monitors and manages agent-available resources."""

class ResourceRegistry:
    def register_resource(self, resource: ResourceRef):
        ...

    def check_availability(self, resource_id: str) -> bool:
        """Check if a resource is currently available."""
        ...

    def health_check_all(self):
        """Periodic health check for all registered resources."""
        ...

    def get_available(self, resource_type: ResourceType | None = None) -> list[ResourceRef]:
        ...


# backend/world/context_manager.py
"""ContextManager: save/restore agent execution context across task switches."""

class ContextManager:
    def save_context(self, tenant_id: int, user_id: int, snapshot: ContextSnapshot):
        """Save current execution context for later resumption."""
        ...

    def restore_context(self, context_id: str) -> ContextSnapshot | None:
        """Restore a previously saved execution context."""
        ...

    def list_open_contexts(self, tenant_id: int, user_id: int) -> list[ContextSnapshot]:
        """List all contexts that can be resumed."""
        ...

    def close_context(self, context_id: str):
        """Mark a context as resolved (task complete or cancelled)."""
        ...
```

---

## Phase 3: Orchestrator Integration

### Task 4: Hook World State into Supervisor

**Files:**
- Modify: `backend/agent/orchestrator.py`
- Modify: `backend/agent/brain.py`

在 supervisor_node 开始时注入 World State:

```python
# v24: Inject World State into supervisor context
from backend.world.state_manager import get_state_manager

state_mgr = get_state_manager()
world_summary = state_mgr.get_status_summary(tenant_id, user_id)

# Prepend to user message for supervisor context
enriched_query = f"{world_summary}\n\n## 用户问题\n{user_query}"
```

在生成 Artifact 时注册:

```python
# v24: Register artifact after generation
from backend.world.artifact_registry import get_artifact_registry

artifact = ArtifactRef(
    artifact_id=f"art_{uuid.uuid4().hex[:12]}",
    artifact_type=ArtifactType.REPORT,
    title=report_title,
    status=ArtifactStatus.COMPLETE,
    file_path=f"/research/{execution_id}/report.md",
)
get_artifact_registry().register(tenant_id, user_id, artifact)
```

新增 SSE event `world_state`:

```python
# backend/agent/brain.py: emit world state update
await emit_event("world_state", {
    "active_tasks": len(state.active_tasks),
    "artifacts": len(state.artifacts),
    "resources_available": available_count,
})
```

---

## Phase 4: Research Executor Integration

### Task 5: Hook into Research Executor

**Files:**
- Modify: `backend/research/executor.py`

```python
# v24: Register task in World State when research starts
state_mgr = get_state_manager()
state_mgr.add_task(tenant_id, user_id, execution_id)
state_mgr.update_goal(tenant_id, user_id, plan.goal)

# v24: Register produced artifacts when research completes
for artifact in generated_artifacts:
    get_artifact_registry().register(tenant_id, user_id, artifact)

# v24: Mark task complete
state_mgr.complete_task(tenant_id, user_id, execution_id)
```

---

## Self-Review

| 610 文档 v24 需求 | 覆盖 |
|---|---|
| WorldState 六维模型 | Task 1 (schemas.py) |
| StateManager (单一数据源) | Task 2 (state_manager.py) |
| ArtifactRegistry (制品追踪) | Task 3 (artifact_registry.py) |
| ResourceRegistry (资源管理) | Task 3 (resource_registry.py) |
| ContextManager (上下文保存/恢复) | Task 3 (context_manager.py) |
| Supervisor 注入 World State | Task 4 (orchestrator.py) |
| SSE world_state 事件 | Task 4 (brain.py) |
| Research Executor 注册任务/制品 | Task 5 (executor.py) |
