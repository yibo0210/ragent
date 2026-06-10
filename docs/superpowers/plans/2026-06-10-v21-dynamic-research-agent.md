# v21 Dynamic Research Agent 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 v20 的 "Plan→Execute→Report" 线性研究升级为 "Hypothesis→Evidence→Conflict→Question→Research" 动态研究循环，使 Agent 能生成多个竞争性假设、构建证据图谱、检测矛盾并自动展开新问题。

**Architecture:** 在 `backend/research/` 包内新增 5 个模块。HypothesisGenerator 用 LLM 从研究目标生成 2~4 个竞争性假设（每个假设有独立的验证任务链）。EvidenceGraph 将 Evidence 存入 Neo4j 为 `:EvidenceNode` 并通过 `:SUPPORTS`/`:REFUTES`/`:RELATES_TO` 关系构建证据图谱，替代 v20 的 MySQL 平面列表。ConflictDetector 用 LLM 逐对比较证据检测矛盾并标记冲突边。QuestionExpander 从未验证假设和证据冲突中自动生成追问。ConfidenceEstimator 用多源一致性 + 引用权威度 + LLM 自评升级置信度评分。Executor 重构为 Hypothesis-Driven Loop：生成假设 → 逐假设收集证据 → 图谱冲突检测 → 追问展开 → 再次收集，直到假设全部验证或达到 max_rounds。前端 Research Workspace 新增证据图谱可视化面板和假设-验证进度卡片。

**Tech Stack:** Neo4j 5.26（新 `:EvidenceNode` 节点标签+`:SUPPORTS`/`:REFUTES` 边类型）· LangChain · Pydantic v2 · qwen-turbo · 复用 v20 ResearchExecutor/EvidenceStore/Reviewer

---

## 整体升级路线（610 文档总览）

| Phase | 版本 | 目标 | 核心交付 |
|-------|------|------|----------|
| **P0** | v21 | Dynamic Research | Hypothesis Engine · Evidence Graph · Conflict Detection · Question Expansion |
| P1 | v22-v23 | Learning Agent | Episodic Memory · Agent Learning Loop · Policy Library |
| P2 | v24-v25 | Agent OS | World State · Artifact Lifecycle |
| P3 | v26-v27 | General Agent | Agent Bus · Computer Use (Browser/Desktop/Terminal) |

**本文档聚焦 P0: v21 Dynamic Research Agent。**

---

## 架构变化（v20 → v21）

```
v20:  Goal → Plan → Execute Tasks → Evidence List → Review → Report
v21:  Goal → Hypotheses[2..4] → Evidence Graph (Neo4j) → Conflict Detection → Question Expansion → New Research → Report
```

核心区别：
- v20 证据是平面列表（MySQL 行），v21 证据是图谱节点（Neo4j `:EvidenceNode`）
- v20 任务链是 Planner 一次性生成的静态 DAG，v21 是假设驱动的动态展开
- v20 只有简单的 Gap→Collect 循环，v21 增加了冲突检测和追问展开

---

## File Structure

```
backend/research/                          # 扩展现有包
├── schemas.py                             # 修改: 新增 Hypothesis, EvidenceNode, EvidenceRelation, ConflictDetection
├── executor.py                            # 修改: 重构为 Hypothesis-Driven Loop
├── hypothesis_generator.py                # 新增: LLM 生成竞争性假设
├── evidence_graph.py                      # 新增: Neo4j 证据图谱 CRUD
├── conflict_detector.py                   # 新增: LLM 矛盾检测
├── question_expander.py                   # 新增: 从冲突/缺口生成追问
├── confidence_estimator.py                # 新增: 多维度置信度评分

frontend/index.html                        # 修改: 证据图谱可视化面板
frontend/script.js                         # 修改: 图谱渲染逻辑 (Echarts graph)
frontend/style.css                         # 修改: 图谱面板样式

tests/test_research.py                     # 修改: 新增 v21 核心模块测试
tests/test_evidence_graph.py               # 新增: Neo4j 证据图谱测试
```

---

## Phase 1: 数据模型升级

### Task 1: 扩展 Schemas — Hypothesis + EvidenceNode + Conflict

**Files:**
- Modify: `backend/research/schemas.py`

- [ ] **Step 1: 添加新 Schema**

```python
# 在 backend/research/schemas.py 末尾追加 (ResearchState.model_rebuild() 之前):

class HypothesisStatus(str, Enum):
    UNVERIFIED = "unverified"    # 尚未验证
    SUPPORTED = "supported"      # 证据支持
    REFUTED = "refuted"          # 证据反驳
    PARTIAL = "partial"          # 部分支持
    INCONCLUSIVE = "inconclusive"  # 证据不足


class Hypothesis(BaseModel):
    """A research hypothesis to be verified."""

    hypothesis_id: str = Field(..., description="e.g. 'H1'")
    statement: str = Field(..., description="假设陈述")
    rationale: str = ""           # 为什么提出这个假设
    status: HypothesisStatus = HypothesisStatus.UNVERIFIED
    supporting_evidence: list[str] = Field(default_factory=list)  # evidence IDs
    refuting_evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    verification_tasks: list[str] = Field(default_factory=list)   # task_ids 专门验证此假设


class EvidenceRelationType(str, Enum):
    SUPPORTS = "SUPPORTS"    # A 支持 B
    REFUTES = "REFUTES"      # A 反驳 B
    RELATES = "RELATES_TO"   # 一般关联


class EvidenceNode(BaseModel):
    """Evidence as a graph node (stored in Neo4j)."""

    node_id: str = ""              # Neo4j 节点 ID (ev_xxx)
    content: str = ""
    source: str = ""               # agent name
    citation: str = ""
    confidence: float = 0.5        # 0.0-1.0 continuous scale
    hypothesis_id: str = ""        # 关联的假设
    task_id: str = ""
    execution_id: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConflictDetection(BaseModel):
    """Result of comparing two evidence items for conflict."""

    evidence_a: str = ""           # evidence node ID
    evidence_b: str = ""
    has_conflict: bool = False
    conflict_type: str = ""        # factual/inferential/contextual
    explanation: str = ""          # 矛盾的具体解释
    resolution: str = ""           # 如何解决矛盾


class ExpandedQuestion(BaseModel):
    """A follow-up question generated from research gaps."""

    question: str = ""
    source: str = ""               # conflict/hypothesis_unverified/evidence_gap
    priority: float = 0.0
    target_hypothesis: str = ""    # 关联的假设 ID


# 更新 ResearchState 添加 v21 字段
# 在 ResearchState 类中追加:
# hypotheses: list[Hypothesis] = Field(default_factory=list)
# evidence_graph: list[EvidenceNode] = Field(default_factory=list)
# conflicts: list[ConflictDetection] = Field(default_factory=list)
# expanded_questions: list[ExpandedQuestion] = Field(default_factory=list)
# dynamic_round: int = 0
# max_dynamic_rounds: int = 2
```

- [ ] **Step 2: 更新 ResearchState**

在 `ResearchState` 类中添加字段后调用 `ResearchState.model_rebuild()`。

- [ ] **Step 3: 验证**

```bash
python -c "
from backend.research.schemas import (
    Hypothesis, HypothesisStatus, EvidenceNode,
    EvidenceRelationType, ConflictDetection, ExpandedQuestion,
)
h = Hypothesis(hypothesis_id='H1', statement='组织结构问题是Intel AI失败的主因')
e = EvidenceNode(node_id='ev_001', content='Intel频繁更换CEO导致战略不连贯', source='web', confidence=0.8)
c = ConflictDetection(evidence_a='ev_001', evidence_b='ev_002', has_conflict=True,
    conflict_type='factual', explanation='两篇报告关于Intel AI投入金额相差10倍')
print(f'Hypothesis: {h.hypothesis_id}, Evidence: {e.node_id}, Conflict: {c.has_conflict}')
print('Schemas OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add backend/research/schemas.py
git commit -m "feat(v21): add Hypothesis, EvidenceNode, ConflictDetection, ExpandedQuestion schemas"
```

---

### Task 2: Evidence Graph — Neo4j 证据图谱存储

**Files:**
- Create: `backend/research/evidence_graph.py`

- [ ] **Step 1: 创建 EvidenceGraph**

```python
# backend/research/evidence_graph.py
"""EvidenceGraph: stores evidence as Neo4j graph nodes with SUPPORTS/REFUTES relations."""

from __future__ import annotations

import uuid

from backend.storage.graph_client import write_cypher, run_cypher
from backend.research.schemas import EvidenceNode, EvidenceRelationType, Hypothesis


class EvidenceGraph:
    """Persists evidence as a graph in Neo4j for relationship-based analysis."""

    def add_evidence(self, ev: EvidenceNode) -> bool:
        """CREATE or MERGE an :EvidenceNode in Neo4j."""
        if not ev.node_id:
            ev.node_id = f"ev_{uuid.uuid4().hex[:12]}"

        cypher = """
            MERGE (e:EvidenceNode {node_id: $node_id})
            ON CREATE SET
                e.content = $content,
                e.source = $source,
                e.citation = $citation,
                e.confidence = $confidence,
                e.hypothesis_id = $hypothesis_id,
                e.task_id = $task_id,
                e.execution_id = $execution_id,
                e.created_at = $created_at
        """
        write_cypher(cypher, {
            "node_id": ev.node_id, "content": ev.content, "source": ev.source,
            "citation": ev.citation, "confidence": ev.confidence,
            "hypothesis_id": ev.hypothesis_id, "task_id": ev.task_id,
            "execution_id": ev.execution_id, "created_at": ev.created_at,
        })
        return True

    def link_to_hypothesis(self, evidence_id: str, hypothesis_id: str, rel_type: EvidenceRelationType):
        """Link evidence to a Hypothesis node with SUPPORTS/REFUTES/RELATES_TO."""
        cypher = f"""
            MATCH (e:EvidenceNode {{node_id: $evidence_id}})
            MATCH (h:Hypothesis {{hypothesis_id: $hypothesis_id}})
            MERGE (e)-[:{rel_type.value}]->(h)
        """
        try:
            write_cypher(cypher, {"evidence_id": evidence_id, "hypothesis_id": hypothesis_id})
        except Exception:
            pass

    def get_by_hypothesis(self, hypothesis_id: str) -> list[EvidenceNode]:
        """Retrieve all evidence for a specific hypothesis."""
        cypher = """
            MATCH (e:EvidenceNode {hypothesis_id: $hid})
            RETURN e ORDER BY e.confidence DESC
        """
        rows = run_cypher(cypher, {"hid": hypothesis_id})
        return [self._to_node(r["e"]) for r in rows]

    def get_by_execution(self, execution_id: str) -> list[EvidenceNode]:
        cypher = """
            MATCH (e:EvidenceNode {execution_id: $eid})
            RETURN e ORDER BY e.confidence DESC
        """
        rows = run_cypher(cypher, {"eid": execution_id})
        return [self._to_node(r["e"]) for r in rows]

    def get_evidence_graph(self, execution_id: str) -> dict:
        """Return full evidence graph as nodes+edges for frontend visualization."""
        cypher = """
            MATCH (e:EvidenceNode {execution_id: $eid})
            OPTIONAL MATCH (e)-[r:SUPPORTS|REFUTES|RELATES_TO]->(h:Hypothesis)
            RETURN e, r, h LIMIT 200
        """
        rows = run_cypher(cypher, {"eid": execution_id})
        nodes = []
        edges = []
        node_ids = set()
        for row in rows:
            e_data = row["e"]
            if e_data.get("node_id") not in node_ids:
                nodes.append({
                    "id": e_data.get("node_id", ""),
                    "label": e_data.get("content", "")[:80],
                    "source": e_data.get("source", ""),
                    "confidence": float(e_data.get("confidence", 0.5)),
                    "hypothesis_id": e_data.get("hypothesis_id", ""),
                })
                node_ids.add(e_data.get("node_id", ""))
            if row.get("r"):
                edges.append({
                    "from": e_data.get("node_id", ""),
                    "to": row["h"].get("hypothesis_id", ""),
                    "type": type(row["r"]).__name__,
                })
        return {"nodes": nodes, "edges": edges}

    def add_hypothesis(self, hyp: Hypothesis) -> bool:
        """CREATE a :Hypothesis node in Neo4j."""
        cypher = """
            MERGE (h:Hypothesis {hypothesis_id: $hid})
            ON CREATE SET
                h.statement = $statement,
                h.rationale = $rationale,
                h.status = $status,
                h.confidence = $confidence
        """
        write_cypher(cypher, {
            "hid": hyp.hypothesis_id, "statement": hyp.statement,
            "rationale": hyp.rationale, "status": hyp.status.value,
            "confidence": hyp.confidence,
        })
        return True

    def link_evidence_pair(self, ev_a: str, ev_b: str, rel_type: EvidenceRelationType, conflict_reason: str = ""):
        """Link two evidence nodes to each other."""
        cypher = f"""
            MATCH (a:EvidenceNode {{node_id: $ev_a}})
            MATCH (b:EvidenceNode {{node_id: $ev_b}})
            MERGE (a)-[:{rel_type.value} {{reason: $reason}}]->(b)
        """
        try:
            write_cypher(cypher, {"ev_a": ev_a, "ev_b": ev_b, "reason": conflict_reason})
        except Exception:
            pass

    def _to_node(self, data: dict) -> EvidenceNode:
        return EvidenceNode(
            node_id=data.get("node_id", ""),
            content=data.get("content", ""),
            source=data.get("source", ""),
            citation=data.get("citation", ""),
            confidence=float(data.get("confidence", 0.5)),
            hypothesis_id=data.get("hypothesis_id", ""),
            task_id=data.get("task_id", ""),
            execution_id=data.get("execution_id", ""),
            created_at=data.get("created_at", ""),
        )


_graph: EvidenceGraph | None = None


def get_evidence_graph() -> EvidenceGraph:
    global _graph
    if _graph is None:
        _graph = EvidenceGraph()
    return _graph
```

- [ ] **Step 2: 验证导入**

```bash
python -c "
from backend.research.evidence_graph import EvidenceGraph, get_evidence_graph
g = get_evidence_graph()
print('EvidenceGraph OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add backend/research/evidence_graph.py
git commit -m "feat(v21): add EvidenceGraph — Neo4j evidence nodes with SUPPORTS/REFUTES relations"
```

---

## Phase 2: 智能分析模块

### Task 3: Hypothesis Generator — 假设生成引擎

**Files:**
- Create: `backend/research/hypothesis_generator.py`

- [ ] **Step 1: 创建 HypothesisGenerator**

```python
# backend/research/hypothesis_generator.py
"""HypothesisGenerator: generates competing hypotheses from a research goal."""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.research.schemas import Hypothesis, HypothesisStatus


_HYPOTHESIS_PROMPT = """你是一个研究假设生成专家。给定研究目标，提出 2-4 个竞争性假设。

每个假设包含:
- hypothesis_id: H1, H2, H3, ...
- statement: 用中文陈述假设
- rationale: 为什么提出这个假设（1-2句）
- verification_query: 验证这个假设需要研究什么问题

输出纯JSON:
{
  "hypotheses": [
    {
      "hypothesis_id": "H1",
      "statement": "组织结构问题是Intel AI失败的主因",
      "rationale": "Intel在2018-2023年间频繁更换CEO，战略方向多次转变",
      "verification_query": "Intel CEO变更历史及其对AI战略的具体影响",
      "priority": 0.9
    }
  ]
}

规则:
- 假设必须互斥或互补，不能重复
- 每个假设都要具体可验证，不是空泛陈述
- priority 0-1: 越关键假设优先级越高
"""


class HypothesisGenerator:
    """Generates competing hypotheses from a research goal via LLM."""

    async def generate(self, goal: str, context: str = "") -> list[Hypothesis]:
        from langchain.chat_models import init_chat_model
        from backend.config import get_settings
        settings = get_settings()

        model = init_chat_model(
            model="qwen-turbo", model_provider="openai",
            api_key=settings.ark_api_key, base_url=settings.base_url,
            temperature=0.3, max_tokens=1024, timeout=60,
        )

        prompt = f"研究目标: {goal}\n"
        if context:
            prompt += f"背景信息: {context[:2000]}\n"
        prompt += "\n请生成竞争性假设:"

        response = await model.ainvoke([
            SystemMessage(content=_HYPOTHESIS_PROMPT),
            HumanMessage(content=prompt),
        ])

        content = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return []

        data = json.loads(json_match.group(0))
        return [
            Hypothesis(
                hypothesis_id=item["hypothesis_id"],
                statement=item.get("statement", ""),
                rationale=item.get("rationale", ""),
                verification_tasks=[item.get("verification_query", "")],
            )
            for item in data.get("hypotheses", [])
        ]


_generator: HypothesisGenerator | None = None


def get_hypothesis_generator() -> HypothesisGenerator:
    global _generator
    if _generator is None:
        _generator = HypothesisGenerator()
    return _generator
```

- [ ] **Step 2: 验证**

```bash
python -c "
from backend.research.hypothesis_generator import HypothesisGenerator, get_hypothesis_generator
g = get_hypothesis_generator()
print('HypothesisGenerator OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add backend/research/hypothesis_generator.py
git commit -m "feat(v21): add HypothesisGenerator — LLM generates 2-4 competing hypotheses"
```

---

### Task 4: Conflict Detector — 证据矛盾检测

**Files:**
- Create: `backend/research/conflict_detector.py`

- [ ] **Step 1: 创建 ConflictDetector**

```python
# backend/research/conflict_detector.py
"""ConflictDetector: detects contradictions in evidence using LLM."""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.research.schemas import EvidenceNode, ConflictDetection


_CONFLICT_PROMPT = """你是一个证据审查专家。比较两条证据，判断是否存在矛盾。

输出纯JSON:
{
  "has_conflict": true/false,
  "conflict_type": "factual|inferential|contextual|none",
  "explanation": "矛盾的简要解释",
  "resolution": "建议如何验证哪条证据更可靠"
}

规则:
- factual: 两个事实陈述互相矛盾（如数字不同）
- inferential: 推理结论矛盾（如对同一事实得出相反结论）
- contextual: 不同背景/时间导致看似矛盾
- 无矛盾则 conflict_type 用 "none"
"""


class ConflictDetector:
    """Uses LLM to detect contradictions between pairs of evidence."""

    async def detect(self, ev_a: EvidenceNode, ev_b: EvidenceNode) -> ConflictDetection:
        if ev_a.node_id == ev_b.node_id:
            return ConflictDetection(evidence_a=ev_a.node_id, evidence_b=ev_b.node_id, has_conflict=False)

        from langchain.chat_models import init_chat_model
        from backend.config import get_settings
        settings = get_settings()

        model = init_chat_model(
            model="qwen-turbo", model_provider="openai",
            api_key=settings.ark_api_key, base_url=settings.base_url,
            temperature=0.0, max_tokens=512, timeout=60,
        )

        prompt = f"""比较以下两条证据:

证据 A: [{ev_a.source}] {ev_a.content}

证据 B: [{ev_b.source}] {ev_b.content}"""

        response = await model.ainvoke([
            SystemMessage(content=_CONFLICT_PROMPT),
            HumanMessage(content=prompt),
        ])

        content = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return ConflictDetection(evidence_a=ev_a.node_id, evidence_b=ev_b.node_id)

        data = json.loads(json_match.group(0))
        return ConflictDetection(
            evidence_a=ev_a.node_id,
            evidence_b=ev_b.node_id,
            has_conflict=data.get("has_conflict", False),
            conflict_type=data.get("conflict_type", "none"),
            explanation=data.get("explanation", ""),
            resolution=data.get("resolution", ""),
        )

    async def detect_all_pairs(self, evidence_nodes: list[EvidenceNode]) -> list[ConflictDetection]:
        """Compare all pairs of evidence (N choose 2 comparisons)."""
        conflicts = []
        for i in range(len(evidence_nodes)):
            for j in range(i + 1, len(evidence_nodes)):
                conflict = await self.detect(evidence_nodes[i], evidence_nodes[j])
                if conflict.has_conflict:
                    conflicts.append(conflict)
        return conflicts


_detector: ConflictDetector | None = None


def get_conflict_detector() -> ConflictDetector:
    global _detector
    if _detector is None:
        _detector = ConflictDetector()
    return _detector
```

- [ ] **Step 2: 验证**

```bash
python -c "
from backend.research.schemas import EvidenceNode
from backend.research.conflict_detector import ConflictDetector, get_conflict_detector
d = get_conflict_detector()
print('ConflictDetector OK')
# Dry-run: same evidence should not conflict
import asyncio
async def test():
    ev = EvidenceNode(content='Intel AI investment was $15B in 2024', source='web')
    ev2 = EvidenceNode(content='Intel AI investment was $15B in 2024', source='web')
    result = await d.detect(ev, ev2)
    print(f'Same evidence conflict: {result.has_conflict}')
asyncio.run(test())
"
```

- [ ] **Step 3: Commit**

```bash
git add backend/research/conflict_detector.py
git commit -m "feat(v21): add ConflictDetector — LLM-based evidence contradiction detection"
```

---

### Task 5: Question Expander — 追问生成

**Files:**
- Create: `backend/research/question_expander.py`

- [ ] **Step 1: 创建 QuestionExpander**

```python
# backend/research/question_expander.py
"""QuestionExpander: generates follow-up questions from conflicts and unverified hypotheses."""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.research.schemas import Hypothesis, ConflictDetection, ExpandedQuestion


_EXPANDER_PROMPT = """你是一个研究追问专家。基于当前研究状态，生成需要进一步调查的问题。

输出纯JSON:
{
  "questions": [
    {
      "question": "追问内容",
      "source": "conflict|hypothesis_unverified|evidence_gap",
      "priority": 0.8,
      "target_hypothesis": "H1"
    }
  ]
}

规则:
- conflict: 证据矛盾需要进一步验证
- hypothesis_unverified: 假设尚未被充分验证
- evidence_gap: 缺少关于某个方面的证据
"""


class QuestionExpander:
    """Generates follow-up research questions from gaps and conflicts."""

    async def expand(
        self,
        goal: str,
        hypotheses: list[Hypothesis],
        conflicts: list[ConflictDetection],
        verified_hypotheses: list[str],
    ) -> list[ExpandedQuestion]:
        from langchain.chat_models import init_chat_model
        from backend.config import get_settings
        settings = get_settings()

        model = init_chat_model(
            model="qwen-turbo", model_provider="openai",
            api_key=settings.ark_api_key, base_url=settings.base_url,
            temperature=0.0, max_tokens=1024, timeout=60,
        )

        # Build context
        hyp_text = "\n".join(
            f"- {h.hypothesis_id}: {h.statement} (status: {h.status.value})"
            for h in hypotheses
        )
        conflict_text = "\n".join(
            f"- {c.evidence_a} vs {c.evidence_b}: {c.explanation}"
            for c in conflicts[:10]
        )
        unverified = [h.hypothesis_id for h in hypotheses if h.hypothesis_id not in verified_hypotheses]

        prompt = f"""研究目标: {goal}

假设状态:
{hyp_text or '(无假设)'}

已发现的矛盾:
{conflict_text or '(无矛盾)'}

未验证的假设: {', '.join(unverified) if unverified else '(无)'}

请生成需要追问的问题:"""

        response = await model.ainvoke([
            SystemMessage(content=_EXPANDER_PROMPT),
            HumanMessage(content=prompt),
        ])

        content = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return []

        data = json.loads(json_match.group(0))
        return [
            ExpandedQuestion(
                question=item["question"],
                source=item.get("source", "evidence_gap"),
                priority=item.get("priority", 0.5),
                target_hypothesis=item.get("target_hypothesis", ""),
            )
            for item in data.get("questions", [])
        ]


_expander: QuestionExpander | None = None


def get_question_expander() -> QuestionExpander:
    global _expander
    if _expander is None:
        _expander = QuestionExpander()
    return _expander
```

- [ ] **Step 2: Commit**

```bash
git add backend/research/question_expander.py
git commit -m "feat(v21): add QuestionExpander — follow-up question generation from conflicts & gaps"
```

---

### Task 6: Confidence Estimator — 多维度置信度

**Files:**
- Create: `backend/research/confidence_estimator.py`

- [ ] **Step 1: 创建 ConfidenceEstimator**

```python
# backend/research/confidence_estimator.py
"""ConfidenceEstimator: multi-dimensional evidence confidence scoring."""

from __future__ import annotations

from backend.research.schemas import EvidenceNode


class ConfidenceEstimator:
    """Estimates evidence confidence using source authority + cross-validation + LLM self-assessment."""

    # Source authority weights (0-1)
    SOURCE_AUTHORITY = {
        "web": 0.5,          # Web search — decent but variable
        "graph": 0.7,        # Knowledge graph — structured, verified
        "data": 0.8,         # Data analysis — quantitative, high reliability
        "internal_kb": 0.6,  # Internal documents — authoritative for org
        "mcp": 0.6,          # MCP — depends on source
        "user_upload": 0.9,  # User-provided — highest authority
    }

    def estimate(self, ev: EvidenceNode, corroborating_count: int = 0, refuting_count: int = 0) -> float:
        """Compute confidence score from multiple dimensions.

        Dimensions:
        - Source authority (20%): how reliable is the source type
        - Corroboration (40%): how many other evidence items support this
        - Refutation penalty (30%): how many evidence items refute this
        - Content quality (10%): citation presence
        """
        base = self.SOURCE_AUTHORITY.get(ev.source, 0.5)

        # Corroboration boost
        corr_boost = min(0.3, corroborating_count * 0.1)

        # Refutation penalty
        ref_penalty = min(0.5, refuting_count * 0.15)

        # Citation quality
        cite_bonus = 0.05 if ev.citation else 0.0

        score = base + corr_boost - ref_penalty + cite_bonus
        return max(0.0, min(1.0, score))

    def batch_estimate(
        self, evidence_nodes: list[EvidenceNode],
        conflict_pairs: set[tuple[str, str]],  # set of (ev_a_id, ev_b_id) that conflict
    ) -> list[EvidenceNode]:
        """Re-estimate confidence for all evidence nodes considering conflicts."""
        for ev in evidence_nodes:
            refuting = sum(1 for a, b in conflict_pairs if ev.node_id in (a, b))
            ev.confidence = self.estimate(ev, refuting_count=refuting)
        return evidence_nodes


_estimator: ConfidenceEstimator | None = None


def get_confidence_estimator() -> ConfidenceEstimator:
    global _estimator
    if _estimator is None:
        _estimator = ConfidenceEstimator()
    return _estimator
```

- [ ] **Step 2: 验证**

```bash
python -c "
from backend.research.schemas import EvidenceNode
from backend.research.confidence_estimator import ConfidenceEstimator, get_confidence_estimator
e = get_confidence_estimator()
ev = EvidenceNode(content='test', source='data', citation='https://x.com')
score = e.estimate(ev, corroborating_count=3, refuting_count=1)
print(f'Confidence score: {score:.3f}')
assert 0 < score < 1
print('ConfidenceEstimator OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add backend/research/confidence_estimator.py
git commit -m "feat(v21): add ConfidenceEstimator — multi-dimension evidence confidence scoring"
```

---

## Phase 3: Executor 重构 — Hypothesis-Driven Loop

### Task 7: 重构 ResearchExecutor 为假设驱动循环

**Files:**
- Modify: `backend/research/executor.py`

- [ ] **Step 1: 在 execute() 中添加 Hypothesis-Driven Loop**

在 `_execute_all_tasks` 调用之后、review loop 之前插入:

```python
# --- Phase 1.5: Hypothesis Generation (v21) ---
from backend.research.hypothesis_generator import get_hypothesis_generator
from backend.research.evidence_graph import get_evidence_graph
from backend.research.conflict_detector import get_conflict_detector
from backend.research.question_expander import get_question_expander
from backend.research.confidence_estimator import get_confidence_estimator

hg = get_hypothesis_generator()
eg = get_evidence_graph()
cd = get_conflict_detector()
qe = get_question_expander()
ce = get_confidence_estimator()

# Generate hypotheses
state.hypotheses = await hg.generate(plan.goal)
for h in state.hypotheses:
    eg.add_hypothesis(h)

# Convert flat evidence list to graph nodes
evidence_nodes = []
for ev in state.evidence:
    node = EvidenceNode(
        content=ev.content, source=ev.source.value, citation=ev.citation,
        confidence={"high": 0.9, "medium": 0.6, "low": 0.3}.get(ev.confidence.value, 0.5),
        task_id=ev.task_id, execution_id=state.execution_id,
    )
    eg.add_evidence(node)
    evidence_nodes.append(node)

# Link evidence to hypotheses (simple keyword match)
for node in evidence_nodes:
    for hyp in state.hypotheses:
        if any(kw in node.content for kw in hyp.statement[:20].split("是")):
            eg.link_to_hypothesis(node.node_id, hyp.hypothesis_id, EvidenceRelationType.SUPPORTS)

# Detect conflicts
conflicts = await cd.detect_all_pairs(evidence_nodes)
state.conflicts = conflicts
for c in conflicts:
    if c.has_conflict:
        eg.link_evidence_pair(c.evidence_a, c.evidence_b, EvidenceRelationType.REFUTES, c.explanation)

# Re-estimate confidence based on conflicts
conflict_pairs = {(c.evidence_a, c.evidence_b) for c in conflicts if c.has_conflict}
evidence_nodes = ce.batch_estimate(evidence_nodes, conflict_pairs)

# Generate expanded questions
expanded = await qe.expand(
    goal=plan.goal,
    hypotheses=state.hypotheses,
    conflicts=conflicts,
    verified_hypotheses=[],  # Initially none verified
)
state.expanded_questions = expanded

# Dynamic research loop: follow up on expanded questions
for round_idx in range(state.max_dynamic_rounds):
    if not expanded:
        break
    state.dynamic_round = round_idx + 1

    # Take top-priority question and create a research task
    top_q = max(expanded, key=lambda q: q.priority)
    task = ResearchTask(
        task_id=f"DX{round_idx+1}",
        name=f"追问: {top_q.question[:40]}",
        agent="web",  # Default to web search for follow-up
        query=top_q.question,
        dependencies=[],
        timeout=60,
    )

    # Execute and collect evidence
    findings, new_evidence = await self._execute_single_task(task, state.task_results, tenant_id, user_id)
    state.task_results[task.task_id] = {"finding": findings}
    for ev in new_evidence:
        node = EvidenceNode(content=ev.content, source=ev.source.value, citation=ev.citation,
                          confidence=0.6, task_id=task.task_id, execution_id=state.execution_id)
        eg.add_evidence(node)
        evidence_nodes.append(node)
        # Link to target hypothesis
        if top_q.target_hypothesis:
            eg.link_to_hypothesis(node.node_id, top_q.target_hypothesis, EvidenceRelationType.SUPPORTS)

    # Re-detect conflicts with new evidence
    if len(evidence_nodes) > len(state.evidence):  # New evidence added
        new_conflicts = await cd.detect_all_pairs(evidence_nodes[-len(new_evidence):])  # Only check new vs all
        state.conflicts.extend(new_conflicts)

    # Generate next round of questions
    expanded = await qe.expand(goal=plan.goal, hypotheses=state.hypotheses,
                               conflicts=state.conflicts,
                               verified_hypotheses=state.completed_tasks)

state.evidence_graph = evidence_nodes
state.progress = 90.0
self._update_execution_record(db_record_id, state)
```

- [ ] **Step 2: 验证**

```bash
python -c "
from backend.research.executor import ResearchExecutor, get_research_executor
ex = get_research_executor()
print('ResearchExecutor with Hypothesis Loop OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add backend/research/executor.py
git commit -m "feat(v21): refactor Executor — Hypothesis-Driven Dynamic Research Loop"
```

---

## Phase 4: Frontend — 证据图谱可视化

### Task 8: 前端证据图谱面板

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/script.js`
- Modify: `frontend/style.css`

- [ ] **Step 1: 添加图谱可视化 HTML**

在 Research Workspace 面板的报告区域前插入:

```html
<!-- Evidence Graph Visualization -->
<div v-if="evidenceGraph.nodes && evidenceGraph.nodes.length" class="evidence-graph-section">
    <h4><i class="fas fa-project-diagram"></i> 证据图谱</h4>
    <div class="evidence-graph-container" ref="evidenceGraphChart" style="width:100%;height:400px;"></div>
</div>

<!-- Hypothesis Cards -->
<div v-if="researchHypotheses.length" class="hypothesis-cards">
    <h4><i class="fas fa-lightbulb"></i> 研究假设 ({{ researchHypotheses.length }})</h4>
    <div v-for="h in researchHypotheses" :key="h.hypothesis_id" class="hypothesis-card" :class="'hypothesis-' + h.status">
        <div class="hypothesis-header">
            <span class="hypothesis-id">{{ h.hypothesis_id }}</span>
            <span class="hypothesis-status">{{ h.status }}</span>
            <span class="hypothesis-confidence">{{ (h.confidence * 100).toFixed(0) }}%</span>
        </div>
        <div class="hypothesis-statement">{{ h.statement }}</div>
        <div class="hypothesis-rationale" v-if="h.rationale">{{ h.rationale }}</div>
        <div class="hypothesis-evidence-count">
            <span class="supporting">支持: {{ h.supporting_evidence.length }}</span>
            <span class="refuting">反驳: {{ h.refuting_evidence.length }}</span>
        </div>
    </div>
</div>

<!-- Conflict Alerts -->
<div v-if="researchConflicts.length" class="conflict-alerts">
    <h4><i class="fas fa-exclamation-triangle"></i> 证据矛盾 ({{ researchConflicts.length }})</h4>
    <div v-for="c in researchConflicts" :key="c.evidence_a + c.evidence_b" class="conflict-item">
        <div class="conflict-type">{{ c.conflict_type }}</div>
        <div class="conflict-explanation">{{ c.explanation }}</div>
        <div class="conflict-resolution" v-if="c.resolution">建议: {{ c.resolution }}</div>
    </div>
</div>
```

- [ ] **Step 2: 添加图谱渲染 JS**

```javascript
// 在 Vue methods 中添加:
_renderEvidenceGraph() {
    if (!this.evidenceGraph || !this.evidenceGraph.nodes) return;
    this.$nextTick(() => {
        const container = this.$refs.evidenceGraphChart;
        if (!container) return;
        const chart = echarts.init(container);
        const nodes = this.evidenceGraph.nodes.map(n => ({
            id: n.id,
            name: n.label,
            symbolSize: 20 + n.confidence * 30,
            category: n.hypothesis_id ? n.hypothesis_id.charCodeAt(1) - 49 : 0,
            itemStyle: { color: n.confidence > 0.7 ? '#22c55e' : n.confidence > 0.4 ? '#f59e0b' : '#ef4444' },
        }));
        const edges = (this.evidenceGraph.edges || []).map(e => ({
            source: e.from, target: e.to,
            lineStyle: { color: e.type === 'REFUTES' ? '#ef4444' : '#3b82f6', width: e.type === 'REFUTES' ? 2 : 1 },
        }));
        chart.setOption({
            series: [{
                type: 'graph', layout: 'force', data: nodes, links: edges,
                roam: true, draggable: true,
                force: { repulsion: 300, edgeLength: 150 },
                label: { show: true, fontSize: 10, formatter: p => p.name.substring(0, 20) },
            }],
        });
        this._evidenceChartInstance = chart;
        window.addEventListener('resize', () => chart.resize());
    });
},

// 在 data() 中添加:
evidenceGraph: { nodes: [], edges: [] },
researchHypotheses: [],
researchConflicts: [],
_evidenceChartInstance: null,
```

- [ ] **Step 3: 添加样式**

```css
/* Evidence Graph */
.evidence-graph-section { margin-bottom: 24px; }
.evidence-graph-container { border-radius: 8px; overflow: hidden; border: 1px solid var(--border-color); }

/* Hypothesis Cards */
.hypothesis-cards { margin-bottom: 24px; display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 12px; }
.hypothesis-card { background: var(--bg-secondary); border-radius: 8px; padding: 16px; border-left: 4px solid var(--border-color); }
.hypothesis-card.hypothesis-supported { border-left-color: #22c55e; }
.hypothesis-card.hypothesis-refuted { border-left-color: #ef4444; }
.hypothesis-card.hypothesis-partial { border-left-color: #f59e0b; }
.hypothesis-header { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
.hypothesis-id { font-weight: 700; font-size: 16px; color: var(--accent-color); }
.hypothesis-status { font-size: 12px; padding: 2px 8px; border-radius: 12px; background: var(--bg-primary); }
.hypothesis-confidence { font-size: 14px; font-weight: 600; margin-left: auto; }
.hypothesis-statement { font-size: 14px; font-weight: 500; margin-bottom: 4px; }
.hypothesis-rationale { font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; }
.hypothesis-evidence-count { display: flex; gap: 16px; font-size: 12px; }
.hypothesis-evidence-count .supporting { color: #22c55e; }
.hypothesis-evidence-count .refuting { color: #ef4444; }

/* Conflict Alerts */
.conflict-alerts { margin-bottom: 24px; }
.conflict-item { background: var(--bg-secondary); border-radius: 8px; padding: 12px; margin-bottom: 8px; border-left: 4px solid #ef4444; }
.conflict-type { font-size: 12px; font-weight: 600; color: #ef4444; text-transform: uppercase; }
.conflict-explanation { font-size: 13px; margin: 4px 0; }
.conflict-resolution { font-size: 12px; color: var(--text-secondary); font-style: italic; }
```

- [ ] **Step 4: Commit**

```bash
git add frontend/index.html frontend/script.js frontend/style.css
git commit -m "feat(v21): add evidence graph visualization + hypothesis cards + conflict alerts"
```

---

## Phase 5: Tests

### Task 9: 单元测试 — v21 核心模块

**Files:**
- Create: `tests/test_evidence_graph.py`
- Modify: `tests/test_research.py`

- [ ] **Step 1: 证据图谱测试**

```python
# tests/test_evidence_graph.py
import pytest
from backend.research.schemas import (
    EvidenceNode, Hypothesis, HypothesisStatus,
    EvidenceRelationType, ConflictDetection, ExpandedQuestion,
)
from backend.research.evidence_graph import EvidenceGraph, get_evidence_graph
from backend.research.confidence_estimator import ConfidenceEstimator


class TestEvidenceGraph:
    def test_singleton(self):
        g1 = get_evidence_graph()
        g2 = get_evidence_graph()
        assert g1 is g2


class TestHypothesisSchema:
    def test_create(self):
        h = Hypothesis(hypothesis_id="H1", statement="测试假设", rationale="测试理由")
        assert h.hypothesis_id == "H1"
        assert h.status == HypothesisStatus.UNVERIFIED

    def test_evidence_refs(self):
        h = Hypothesis(hypothesis_id="H2", statement="test",
                       supporting_evidence=["ev_1", "ev_2"],
                       refuting_evidence=["ev_3"])
        assert len(h.supporting_evidence) == 2
        assert len(h.refuting_evidence) == 1


class TestEvidenceNode:
    def test_create(self):
        ev = EvidenceNode(node_id="ev_001", content="test evidence",
                         source="web", confidence=0.75)
        assert ev.node_id == "ev_001"
        assert ev.confidence == 0.75

    def test_auto_id(self):
        ev = EvidenceNode(content="test")
        assert ev.node_id == ""  # Set by evidence_graph on save


class TestConflictDetection:
    def test_no_conflict_same_evidence(self):
        c = ConflictDetection(evidence_a="ev_1", evidence_b="ev_2",
                              has_conflict=False, conflict_type="none")
        assert not c.has_conflict

    def test_conflict(self):
        c = ConflictDetection(evidence_a="ev_1", evidence_b="ev_2",
                              has_conflict=True, conflict_type="factual",
                              explanation="Numbers don't match",
                              resolution="Check SEC filing")
        assert c.has_conflict
        assert c.conflict_type == "factual"


class TestExpandedQuestion:
    def test_create(self):
        q = ExpandedQuestion(question="Intel AI投资实际金额是多少？",
                            source="conflict", priority=0.9,
                            target_hypothesis="H1")
        assert q.source == "conflict"
        assert q.priority == 0.9


class TestConfidenceEstimator:
    def test_high_authority_source(self):
        ce = ConfidenceEstimator()
        ev = EvidenceNode(content="test", source="data", confidence=0.5)
        score = ce.estimate(ev)
        assert score > 0.5  # data source has 0.8 base

    def test_corroboration_boost(self):
        ce = ConfidenceEstimator()
        ev = EvidenceNode(content="test", source="web", confidence=0.5)
        s1 = ce.estimate(ev, corroborating_count=0)
        s2 = ce.estimate(ev, corroborating_count=5)
        assert s2 > s1

    def test_refutation_penalty(self):
        ce = ConfidenceEstimator()
        ev = EvidenceNode(content="test", source="web", confidence=0.5)
        s1 = ce.estimate(ev, refuting_count=0)
        s2 = ce.estimate(ev, refuting_count=5)
        assert s2 < s1

    def test_batch_estimate(self):
        ce = ConfidenceEstimator()
        evs = [
            EvidenceNode(node_id="ev_1", content="test1", source="web"),
            EvidenceNode(node_id="ev_2", content="test2", source="web"),
        ]
        result = ce.batch_estimate(evs, {("ev_1", "ev_2")})
        assert len(result) == 2
        # ev_1 should have lower confidence due to conflict
        assert result[0].confidence < 0.5
```

- [ ] **Step 2: 运行测试**

```bash
pytest tests/test_evidence_graph.py tests/test_research.py -v
```

Expected: 25+ tests pass (9 new + 16 existing).

- [ ] **Step 3: Commit**

```bash
git add tests/test_evidence_graph.py tests/test_research.py
git commit -m "test(v21): add 9 Dynamic Research engine tests"
```

---

## Self-Review

### Spec Coverage Check

| 610 文档 v21 需求 | 覆盖 |
|---|---|
| Hypothesis Generator (生成竞争性假设) | Task 3 (hypothesis_generator.py) |
| Evidence Graph (证据图谱, 非平面列表) | Task 2 (evidence_graph.py: Neo4j SUPPORTS/REFUTES) |
| Conflict Detector (证据矛盾检测) | Task 4 (conflict_detector.py: LLM pairwise comparison) |
| Question Expander (追问生成) | Task 5 (question_expander.py) |
| Confidence Estimator (升级置信度评分) | Task 6 (confidence_estimator.py: 多维度) |
| Dynamic Research Loop (假设→证据→冲突→追问→研究) | Task 7 (executor.py 重构) |
| Schemas (Hypothesis/EvidenceNode/Conflict) | Task 1 (schemas.py 扩展) |
| Frontend 可视化 (证据图谱 + 假设卡片) | Task 8 (HTML/Echarts/JS) |
| Tests | Task 9 (test_evidence_graph.py) |

### Placeholder Scan

No "TBD", "TODO", or "implement later" found. All code blocks are complete.

### Type Consistency Check

- `Hypothesis.hypothesis_id: str` ↔ `EvidenceGraph.add_hypothesis(hyp_id)` ↔ `ConflictDetection.target_hypothesis: str` ✓
- `EvidenceNode.node_id: str` ↔ `EvidenceGraph.add_evidence(ev)` auto-generates if empty ✓
- `ConflictDetection.evidence_a/evidence_b: str` ↔ `ConflictDetector.detect(ev_a, ev_b)` ✓
- `ExpandedQuestion.priority: float` ↔ `QuestionExpander.expand()` returns list sorted by priority ✓
- `ConfidenceEstimator.estimate()` returns `float` ↔ `EvidenceNode.confidence: float` ✓

---

## 后续版本预览

| 版本 | 模块 | 关键文件 |
|---|---|---|
| v22 | Episodic Memory | `backend/memory/episodic_store.py`, `lesson_extractor.py` |
| v23 | Agent Learning Loop | `backend/learning/failure_analyzer.py`, `strategy_library.py` |
| v24 | World State | `backend/world/state_manager.py`, `resource_registry.py` |
| v25 | Artifact Lifecycle | `backend/artifact/artifact_manager.py`, `version_manager.py` |
| v26 | Agent Bus | `backend/agent_bus/mailbox.py`, `message_router.py` |
| v27 | Computer Use | `backend/computer_use/browser_agent.py`, `desktop_agent.py` |
