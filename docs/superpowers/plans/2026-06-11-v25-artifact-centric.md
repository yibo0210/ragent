# v25 Artifact-Centric Agent 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for tracking.

**Goal:** 将 Agent 的核心交互模型从"对话驱动"升级为"Artifact 驱动"。Artifact（报告/Excel/Chart/PDF/PPTX）不再是副产物，而是 Agent 的一等公民。引入完整的 Artifact Lifecycle（Create→Version→Update→Review→Publish），支持跨任务引用、版本追溯、增量更新。用户可以说"更新 42 号报告"而不是重新描述需求。

**Architecture:** 新增 `backend/artifact/` 包（独立于 `backend/workflow/artifact.py` 的生成器），包含 4 个模块。ArtifactManager 管理完整的 Artifact CRUD + 状态机转换（draft→review→published→archived）。VersionManager 实现语义版本控制（major.minor.patch），存储每次变更的 diff，支持回滚到任意历史版本。DependencyGraph 用 Neo4j 存储 Artifact 之间的引用/依赖关系（`:DEPENDS_ON`/`:DERIVED_FROM`），构建可追溯的制品血缘。ArtifactMemory 在 Agent 对话中通过 `@artifact_id` 引用制品，使 Agent 能理解"更新上次的报告"等上下文指令。复用 v24 World State 的 ArtifactRegistry 作为发现层。

**Tech Stack:** Neo4j 5.26（Artifact 依赖图）· MySQL 8.0（Artifact 元数据 + 版本历史）· MinIO（Artifact 文件存储）· Pydantic v2 · 复用 v24 ArtifactRegistry · 复用 v16 workflow/artifact.py（生成器）

---

## File Structure

```
backend/artifact/                          # 新增包
├── __init__.py                            # 包入口 + 单例工厂
├── schemas.py                             # Artifact, ArtifactVersion, ArtifactDependency, ArtifactStatus
├── artifact_manager.py                    # CRUD + 状态机 + 引用解析
├── version_manager.py                     # 语义版本 + diff + 回滚
├── dependency_graph.py                    # Neo4j Artifact 血缘图
├── artifact_memory.py                     # 对话上下文中的 Artifact 引用

backend/research/executor.py               # 修改: 生成 Artifact 时注册版本
backend/agent/brain.py                     # 修改: 对话中解析 @artifact_id 引用

tests/test_artifact.py                     # 新增: Artifact 系统测试
```

---

## Phase 1: Schemas + Artifact Manager

### Task 1: Artifact Schemas

**Files:**
- Create: `backend/artifact/__init__.py`
- Create: `backend/artifact/schemas.py`

```python
# backend/artifact/schemas.py
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ArtifactStatus(str, Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class ArtifactType(str, Enum):
    REPORT = "report"
    EXCEL = "excel"
    CHART = "chart"
    PDF = "pdf"
    PPTX = "pptx"
    CSV = "csv"


class VersionBump(str, Enum):
    MAJOR = "major"    # breaking changes / complete rewrite
    MINOR = "minor"    # new sections / data update
    PATCH = "patch"    # typo fixes / formatting


class ArtifactVersion(BaseModel):
    """A specific version of an artifact."""
    version_id: str = ""
    artifact_id: str = ""
    version: str = "1.0.0"                 # semver
    change_summary: str = ""
    change_diff: str = ""                  # short text diff summary (LLM generated)
    file_path: str = ""                    # MinIO object path
    created_by: str = ""                   # user_id or "agent"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ArtifactDependency(BaseModel):
    """A dependency relationship between two artifacts."""
    source_id: str = ""                    # depends on target
    target_id: str = ""                    # is depended on by source
    relation: str = "derived_from"         # derived_from | references | extends


class Artifact(BaseModel):
    """An agent-produced artifact with full lifecycle."""
    artifact_id: str = ""
    title: str = ""
    artifact_type: ArtifactType = ArtifactType.REPORT
    status: ArtifactStatus = ArtifactStatus.DRAFT
    current_version: str = "1.0.0"
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    source_task_id: str = ""               # research execution that created it
    tenant_id: int = 0
    user_id: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
```

### Task 2: Artifact Manager

**Files:**
- Create: `backend/artifact/artifact_manager.py`

```python
# backend/artifact/artifact_manager.py
"""ArtifactManager: full CRUD + state machine for artifacts."""

# State machine transitions
_TRANSITIONS = {
    ArtifactStatus.DRAFT: [ArtifactStatus.IN_REVIEW, ArtifactStatus.ARCHIVED],
    ArtifactStatus.IN_REVIEW: [ArtifactStatus.DRAFT, ArtifactStatus.PUBLISHED, ArtifactStatus.ARCHIVED],
    ArtifactStatus.PUBLISHED: [ArtifactStatus.ARCHIVED],
    ArtifactStatus.ARCHIVED: [ArtifactStatus.DRAFT],  # unarchive
}

class ArtifactManager:
    def create(self, artifact: Artifact) -> Artifact:
        """Create a new artifact in draft status."""
        ...

    def get(self, artifact_id: str) -> Artifact | None:
        ...

    def list_by_tenant(self, tenant_id: int, artifact_type: ArtifactType | None = None) -> list[Artifact]:
        ...

    def update_status(self, artifact_id: str, new_status: ArtifactStatus) -> bool:
        """Transition artifact through lifecycle states."""
        current = self.get(artifact_id)
        if new_status not in _TRANSITIONS[current.status]:
            raise ValueError(f"Cannot transition {current.status} → {new_status}")
        ...

    def search(self, tenant_id: int, query: str, limit: int = 10) -> list[Artifact]:
        """Semantic search across artifact titles/descriptions/tags."""
        ...

    def get_reference_context(self, artifact_id: str) -> str:
        """Format artifact summary for LLM injection when referenced."""
        artifact = self.get(artifact_id)
        if not artifact:
            return f"[未找到制品 {artifact_id}]"
        return (
            f"## 引用制品: {artifact.title} (v{artifact.current_version})\n"
            f"类型: {artifact.artifact_type.value}\n"
            f"描述: {artifact.description}\n"
            f"状态: {artifact.status.value}"
        )
```

---

## Phase 2: Version + Dependency Management

### Task 3: Version Manager + Dependency Graph

**Files:**
- Create: `backend/artifact/version_manager.py`
- Create: `backend/artifact/dependency_graph.py`

```python
# backend/artifact/version_manager.py
"""VersionManager: semantic versioning with diff tracking and rollback."""

class VersionManager:
    def create_version(
        self, artifact_id: str, file_content: str, change_summary: str, bump: VersionBump
    ) -> ArtifactVersion:
        """Save new version, auto-increment semver, generate change diff."""
        ...

    def get_history(self, artifact_id: str) -> list[ArtifactVersion]:
        """Get version history for an artifact (newest first)."""
        ...

    def get_version(self, artifact_id: str, version: str) -> ArtifactVersion | None:
        """Get a specific version."""
        ...

    def rollback(self, artifact_id: str, target_version: str) -> ArtifactVersion:
        """Restore an artifact to a previous version (creates new version)."""
        ...

    def generate_diff_summary(self, old_content: str, new_content: str) -> str:
        """LLM generates a human-readable diff summary."""
        model = init_chat_model("qwen-turbo", max_tokens=512, timeout=60)
        ...


# backend/artifact/dependency_graph.py
"""DependencyGraph: Neo4j-based artifact lineage tracking."""

class DependencyGraph:
    def add_dependency(self, source_id: str, target_id: str, relation: str = "derived_from"):
        """Link two artifacts in Neo4j:
        (a:Artifact {artifact_id: source_id})-[:DEPENDS_ON {relation}]->(b:Artifact {artifact_id: target_id})
        """
        ...

    def get_dependents(self, artifact_id: str) -> list[ArtifactDependency]:
        """Find all artifacts that depend on this one."""
        ...

    def get_dependencies(self, artifact_id: str) -> list[ArtifactDependency]:
        """Find all artifacts this one depends on."""
        ...

    def get_lineage_graph(self, artifact_id: str, depth: int = 3) -> dict:
        """Return full dependency subgraph for visualization (nodes + edges)."""
        ...

    def detect_circular_dependency(self, source_id: str, target_id: str) -> bool:
        """Prevent circular dependency chains."""
        ...

    def get_update_cascade(self, artifact_id: str) -> list[str]:
        """When this artifact is updated, find all dependents that may need review."""
        ...
```

---

## Phase 3: Artifact Memory

### Task 4: Artifact Memory (Contextual References)

**Files:**
- Create: `backend/artifact/artifact_memory.py`

```python
# backend/artifact/artifact_memory.py
"""ArtifactMemory: resolves @artifact references in conversation context."""

import re

_ARTIFACT_REF_PATTERN = re.compile(r'@(\d+)|更新(.+)号报告|制品\s*(\d+)')

class ArtifactMemory:
    def parse_references(self, user_message: str, tenant_id: int) -> list[str]:
        """Extract artifact references from user message.
        Supports: @42, 更新42号报告, 制品 42
        """
        ...

    def inject_artifact_context(self, user_message: str, tenant_id: int) -> str:
        """Resolve all @refs and inject artifact summaries into the message."""
        refs = self.parse_references(user_message, tenant_id)
        if not refs:
            return user_message

        mgr = get_artifact_manager()
        context_parts = []
        for aid in refs:
            ctx = mgr.get_reference_context(aid)
            if ctx:
                context_parts.append(ctx)

        if not context_parts:
            return user_message

        return "\n\n".join(context_parts) + f"\n\n## 当前任务\n{user_message}"
```

---

## Phase 4: Integration

### Task 5: Hook into Research Executor + Brain

**Files:**
- Modify: `backend/research/executor.py`
- Modify: `backend/agent/brain.py`

在研究生成 Artifact 时:

```python
# v25: Register artifact + version when research produces output
from backend.artifact.artifact_manager import get_artifact_manager
from backend.artifact.version_manager import get_version_manager
from backend.artifact.dependency_graph import get_dependency_graph

mgr = get_artifact_manager()
artifact = Artifact(
    artifact_id=f"art_{uuid.uuid4().hex[:12]}",
    title=report_title,
    artifact_type=ArtifactType.REPORT,
    source_task_id=execution_id,
    tenant_id=tenant_id, user_id=user_id,
)
mgr.create(artifact)

ver_mgr = get_version_manager()
ver_mgr.create_version(
    artifact.artifact_id, report_content,
    change_summary="初始版本",
    bump=VersionBump.MINOR,
)
```

在对话中解析 Artifact 引用:

```python
# v25: Resolve @artifact references in chat
from backend.artifact.artifact_memory import get_artifact_memory

# In brain.py chat_with_agent:
enriched_message = get_artifact_memory().inject_artifact_context(message, tenant_id)
```

---

## Self-Review

| 610 文档 v25 需求 | 覆盖 |
|---|---|
| Artifact Lifecycle (Create→Version→Update→Review→Publish) | Task 2 (artifact_manager.py 状态机) |
| Artifact Model (id/type/version/status/dependencies) | Task 1 (schemas.py) |
| ArtifactManager (CRUD + 状态转换) | Task 2 (artifact_manager.py) |
| VersionManager (semver + diff + 回滚) | Task 3 (version_manager.py) |
| DependencyGraph (血缘追踪) | Task 3 (dependency_graph.py) |
| ArtifactMemory (@引用解析) | Task 4 (artifact_memory.py) |
| Research Executor 注册制品 | Task 5 (executor.py) |
| Brain 解析 @引用 | Task 5 (brain.py) |
