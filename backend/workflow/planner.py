"""WorkflowPlanner: decomposes natural language goals into executable DAG plans."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from backend.workflow.schemas import WorkflowStep, WorkflowPlan
from backend.workflow.models import WorkflowDefinition


_PLANNER_SYSTEM_PROMPT = """你是一个工作流规划专家。给定用户的业务目标，输出JSON执行计划。

1. 将目标拆解为具体步骤，每个步骤只调用一个工具。
2. 可用工具:
   - rag_specialist: 知识库搜索（文档、手册、报告）
   - web_searcher: 实时网络搜索（新闻、市场数据、外部信息）
   - data_analyst: SQL数据分析（销售、指标、日志）
   - local_graph_search: 知识图谱实体探索（谁/什么与X相关）
   - global_graph_search: 社区级别摘要（高层次主题聚类）
   - direct_answer: 直接LLM回答（无需检索）

3. 每个步骤必须指定:
   - step_id: 唯一ID（step_1, step_2, ...）
   - name: 简短可读的名称
   - tool: 上述工具名之一
   - query: 具体的中文任务描述
   - dependencies: 必须在前置步骤完成后才能执行的step_id列表
   - input_mapping: 从前置步骤映射变量名
   - timeout: 超时秒数

4. 依赖规则:
   - 无依赖的步骤并行执行
   - 有依赖的步骤等待前置完成
   - 通过依赖串联顺序步骤

5. 合理规划: 数据查询 -> 分析 -> 报告
6. 最多3个步骤，每个步骤的query要尽可能全面，一个步骤涵盖多个维度

只输出有效JSON，不要其他内容:
{
  "steps": [
    {
      "step_id": "step_1",
      "name": "...",
      "tool": "...",
      "query": "...",
      "dependencies": [],
      "input_mapping": {},
      "timeout": 60
    }
  ],
  "reasoning": "分解逻辑"
}
"""


class WorkflowPlanner:
    """Converts natural language goals into WorkflowPlan DAGs."""

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from langchain.chat_models import init_chat_model
            from backend.config import get_settings
            settings = get_settings()
            self._model = init_chat_model(
                model="qwen-turbo", model_provider="openai",
                api_key=settings.ark_api_key, base_url=settings.base_url,
                temperature=0.0, max_tokens=1024, timeout=60,
            )
        return self._model

    async def plan(
        self,
        goal: str,
        tenant_id: int = 0,
        user_id: int = 0,
    ) -> WorkflowPlan:
        """Generate a workflow plan from a user goal."""
        model = self._get_model()

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"目标: {goal}\n\n请生成JSON执行计划。"),
        ]

        response = await model.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

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
                timeout=s.get("timeout", 60),
            ))

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


_planner: WorkflowPlanner | None = None


def get_workflow_planner() -> WorkflowPlanner:
    global _planner
    if _planner is None:
        _planner = WorkflowPlanner()
    return _planner
