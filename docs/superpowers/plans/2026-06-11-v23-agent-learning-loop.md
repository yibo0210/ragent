# v23 Agent Learning Loop 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 v22 Episodic Memory 基础上新增 Agent Learning Loop，使 Agent 能从失败中自动分析根因、提炼策略、积累执行经验，形成持续自我优化的 Execution Policy Library。将现有 Critique→Replan 升级为 Critique→Root Cause→Lesson→Memory→Future Improvement 闭环。

**Architecture:** 新增 `backend/learning/` 包，包含 4 个模块。FailureAnalyzer 拦截 Critique 发现的失败（事实错误、推理漏洞、SQL 错误），通过 LLM 进行根因分析（5 Why 方法），输出 `FailureAnalysis`（root_cause + category + severity）。LessonGenerator 将 FailureAnalysis + Episode 上下文通过 LLM 转化为可执行的 `Policy`（when→then 规则），按 domain（research/sql/retrieval/graph）分类存储。StrategyLibrary 维护 Policy 的 Neo4j 知识库，支持按 domain/context 检索匹配策略，跟踪每条 Policy 的 success_rate + times_applied 统计。PolicyStore 在 Agent 执行关键节点前（SQL 生成、图查询、检索策略选择）注入相关 Policy 作为约束提示，执行后更新 Policy 的统计指标。复用 v19 Memory Graph Store 的 Neo4j 基础设施和 v22 的 Lesson 结构。

**Tech Stack:** Neo4j 5.26（新 `:Policy`/`:FailureAnalysis` 节点标签）· LangChain · Pydantic v2 · qwen-turbo · 复用 v19 MemoryGraphStore · 依赖 v22 Episodic Memory

---

## File Structure

```
backend/learning/                          # 新增包
├── __init__.py                            # 包入口 + 单例工厂
├── schemas.py                             # FailureAnalysis, Policy, PolicyDomain, FailureCategory
├── failure_analyzer.py                    # LLM 根因分析 (5 Why)
├── lesson_generator.py                    # FailureAnalysis + Episode → Policy
├── strategy_library.py                    # Policy Neo4j CRUD + 检索 + 统计
├── policy_store.py                        # 执行上下文注入 + 效果追踪

backend/agent/orchestrator.py              # 修改: critique_node 触发 FailureAnalyzer
backend/research/executor.py               # 修改: 研究完成触发 Policy 效果评估

tests/test_agent_learning.py               # 新增: 学习循环测试
```

---

## Phase 1: Schemas + Failure Analyzer

### Task 1: Learning Schemas

**Files:**
- Create: `backend/learning/__init__.py`
- Create: `backend/learning/schemas.py`

```python
# backend/learning/schemas.py
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class FailureCategory(str, Enum):
    FACTUAL_ERROR = "factual_error"        # 事实错误/幻觉
    REASONING_GAP = "reasoning_gap"        # 推理链条断裂
    SQL_ERROR = "sql_error"                # SQL 生成/执行错误
    RETRIEVAL_FAILURE = "retrieval_failure"  # 检索策略不当
    TOOL_MISUSE = "tool_misuse"            # 工具调用参数错误
    TIMEOUT = "timeout"                    # 超时


class PolicyDomain(str, Enum):
    RESEARCH = "research"
    SQL = "sql"
    RETRIEVAL = "retrieval"
    GRAPH = "graph"
    GENERAL = "general"


class FailureAnalysis(BaseModel):
    analysis_id: str = ""
    episode_id: str = ""
    failure_description: str = ""
    category: FailureCategory = FailureCategory.FACTUAL_ERROR
    root_cause: str = ""                   # 5 Why 根因
    severity: float = 0.5                  # 0-1 严重程度
    context_snapshot: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Policy(BaseModel):
    policy_id: str = ""
    domain: PolicyDomain = PolicyDomain.GENERAL
    name: str = ""                         # 简短名称
    condition: str = ""                    # 触发条件 (when)
    action: str = ""                       # 执行动作 (then)
    rationale: str = ""                    # 原因说明
    source_analysis_ids: list[str] = Field(default_factory=list)
    success_rate: float = 0.0
    times_applied: int = 0
    times_succeeded: int = 0
    importance: float = 0.5
    is_active: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tenant_id: int = 0
```

### Task 2: Failure Analyzer + Lesson Generator

**Files:**
- Create: `backend/learning/failure_analyzer.py`
- Create: `backend/learning/lesson_generator.py`

```python
# backend/learning/failure_analyzer.py
"""FailureAnalyzer: LLM-based root cause analysis using 5 Why method."""

_FIVE_WHY_PROMPT = """分析以下 Agent 失败的根本原因。使用 5 Why 方法逐层追问。

失败信息:
- 任务: {goal}
- 失败描述: {failure}
- 上下文: {context}

输出 JSON:
{{
  "category": "factual_error|reasoning_gap|sql_error|retrieval_failure|tool_misuse|timeout",
  "root_cause": "最根本的原因（一句话）",
  "why_chain": ["原因1", "原因2", "原因3"],
  "severity": 0.7,
  "preventable": true
}}
"""

class FailureAnalyzer:
    async def analyze(
        self,
        failure_description: str,
        goal: str = "",
        context: dict | None = None,
    ) -> FailureAnalysis:
        """Analyze a failure and return root cause analysis."""
        model = init_chat_model("qwen-turbo", max_tokens=1024, timeout=60)
        ...

    def should_analyze(self, critique_result: dict) -> bool:
        """Determine if Critique output warrants failure analysis."""
        return not critique_result.get("is_valid", True)


# backend/learning/lesson_generator.py
"""LessonGenerator: converts FailureAnalysis into actionable Policies."""

_POLICY_PROMPT = """基于以下失败分析，生成可执行的策略规则。

失败分析: {analysis}
历史相关经验: {related_lessons}

输出 JSON:
{{
  "policies": [
    {{
      "domain": "research|sql|retrieval|graph|general",
      "name": "简短策略名",
      "condition": "WHEN 触发条件",
      "action": "THEN 执行动作",
      "rationale": "为什么这个策略有效"
    }}
  ]
}}
"""

class LessonGenerator:
    async def generate_policies(
        self,
        analysis: FailureAnalysis,
        related_lessons: list[str] | None = None,
    ) -> list[Policy]:
        """Generate policies from failure analysis + past lessons."""
        model = init_chat_model("qwen-turbo", max_tokens=1024, timeout=60)
        ...
```

---

## Phase 2: Strategy Library + Policy Store

### Task 3: Strategy Library + Policy Store

**Files:**
- Create: `backend/learning/strategy_library.py`
- Create: `backend/learning/policy_store.py`

```python
# backend/learning/strategy_library.py
"""StrategyLibrary: Neo4j-backed Policy knowledge base."""

class StrategyLibrary:
    def save_policy(self, policy: Policy) -> bool:
        cypher = """
            MERGE (p:Policy {policy_id: $pid})
            ON CREATE SET p.name = $name, p.domain = $domain,
                p.condition = $condition, p.action = $action,
                p.importance = $importance, p.tenant_id = $tid
            ON MATCH SET p.times_applied = p.times_applied + 1
        """
        ...

    def match_policies(
        self, domain: PolicyDomain, context: str, tenant_id: int, limit: int = 5
    ) -> list[Policy]:
        """Retrieve matching policies for a given execution context."""
        ...

    def update_policy_stats(self, policy_id: str, success: bool):
        """Update success_rate after policy application."""
        ...

    def get_top_policies(self, domain: PolicyDomain, tenant_id: int, limit: int = 10) -> list[Policy]:
        ...


# backend/learning/policy_store.py
"""PolicyStore: injects relevant policies into agent execution context."""

class PolicyStore:
    def get_policy_context(self, domain: PolicyDomain, task: str, tenant_id: int) -> str:
        """Format matched policies as LLM prompt injection."""
        lib = get_strategy_library()
        policies = lib.match_policies(domain, task, tenant_id)
        if not policies:
            return ""
        lines = ["## 执行策略提醒 (Agent Learning)"]
        for p in policies:
            lines.append(f"- {p.condition} → {p.action}")
        return "\n".join(lines)

    def record_policy_result(self, policy_id: str, success: bool):
        """Record whether a policy helped. Called after execution."""
        lib = get_strategy_library()
        lib.update_policy_stats(policy_id, success)
```

---

## Phase 3: Orchestrator Integration

### Task 4: Hook into Critique → Replan Loop

**Files:**
- Modify: `backend/agent/orchestrator.py`

在 `critique_node` 检测到失败时触发:

```python
# v23: Hook FailureAnalyzer into critique_node
from backend.learning.failure_analyzer import get_failure_analyzer
from backend.learning.lesson_generator import get_lesson_generator
from backend.learning.strategy_library import get_strategy_library

# In critique_node, when is_valid=False:
analyzer = get_failure_analyzer()
analysis = await analyzer.analyze(
    failure_description=critique_result.feedback,
    goal=state.get("query_plan", {}).get("goal", ""),
    context={"draft_answer": state.get("draft_answer", "")},
)

generator = get_lesson_generator()
policies = await generator.generate_policies(analysis)
lib = get_strategy_library()
for policy in policies:
    lib.save_policy(policy)
```

在需要策略注入的节点前:

```python
# v23: Inject policies before key decisions
from backend.learning.policy_store import get_policy_store

# Before SQL generation in data_analyst:
sql_policies = get_policy_store().get_policy_context(
    PolicyDomain.SQL, user_query, tenant_id
)

# Before retrieval strategy selection:
retrieval_policies = get_policy_store().get_policy_context(
    PolicyDomain.RETRIEVAL, user_query, tenant_id
)
```

---

## Phase 4: Research Executor Integration

### Task 5: Hook into Research Executor

**Files:**
- Modify: `backend/research/executor.py`

在研究任务完成后评估 Policy 效果:

```python
# v23: After research completes, evaluate applied policies
from backend.learning.policy_store import get_policy_store

# If research was successful, credit policies that were applied
if state.status == ResearchTaskStatus.COMPLETED:
    for policy_id in state.applied_policy_ids:
        get_policy_store().record_policy_result(policy_id, success=True)
```

---

## Self-Review

| 610 文档 v23 需求 | 覆盖 |
|---|---|
| Failure Analyzer (5 Why 根因分析) | Task 2 (failure_analyzer.py) |
| Lesson Generator (失败→策略转化) | Task 2 (lesson_generator.py) |
| Strategy Library (Policy 知识库) | Task 3 (strategy_library.py) |
| Policy Store (执行上下文注入) | Task 3 (policy_store.py) |
| Critique Hook (失败→分析→策略) | Task 4 (orchestrator.py) |
| Executor Hook (成功→策略反馈) | Task 5 (executor.py) |
| Execution Policy Library (长期积累) | Phase 2 (StrategyLibrary Neo4j) |
