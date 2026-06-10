# v22 Episodic Memory 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 Semantic Memory（Fact/Preference/Task/Relation）之上新增 Episodic Memory，记录每次研究任务的完整经验（Goal→Plan→Actions→Failures→Outcome→Lessons），使 Agent 能从过去任务中学习并复用经验。

**Architecture:** 在 `backend/memory/` 包内新增 4 个模块。EpisodicStore 将 Episode 以 `:Episode` 节点存入 Neo4j，通过 `:HAS_EPISODE` 关系链接到用户节点。LessonExtractor 在每次研究完成后通过 LLM 从 Episode 中提炼 Lessons（经验教训），Lesson 以 `:Lesson` 节点存储并通过 `:DERIVED_FROM` 链接到 Episode。EpisodeRetriever 在启动新研究时检索相似 Episode 并注入 Planner 上下文。MemoryRanker 根据时效性+相似度+成功/失败权重对历史经验排序。复用 v19 Memory Graph Store 的 Neo4j 基础设施。

**Tech Stack:** Neo4j 5.26（新 `:Episode`/`:Lesson` 节点标签）· LangChain · Pydantic v2 · qwen-turbo · 复用 v19 MemoryGraphStore/v20 ResearchExecutor

---

## File Structure

```
backend/memory/                          # 扩展现有包
├── schemas.py                           # 修改: 新增 Episode, Lesson, LessonType
├── episodic_store.py                    # 新增: Episode/Lesson Neo4j CRUD
├── lesson_extractor.py                  # 新增: LLM 从 Episode 提炼 Lessons
├── episode_retriever.py                 # 新增: 检索相似 Episode
├── memory_ranker.py                     # 新增: 经验排序 (时效+相似+成功权重)

backend/research/executor.py             # 修改: 研究完成后调用 lesson_extractor
backend/research/planner.py              # 修改: 生成计划前注入相关 Episodes

tests/test_episodic_memory.py            # 新增: 情景记忆测试
```

---

## Phase 1: Schemas + Store

### Task 1: Episode/Lesson Schemas

**Files:**
- Modify: `backend/memory/schemas.py`

```python
# 在 backend/memory/schemas.py 末尾追加:

class LessonType(str, Enum):
    STRATEGY = "strategy"      # 策略经验 (e.g., "先查SEC Filing")
    TOOL = "tool"              # 工具使用经验
    PITFALL = "pitfall"        # 陷阱/失败教训
    INSIGHT = "insight"        # 洞察/发现


class Episode(BaseModel):
    """A complete research experience record."""

    episode_id: str = ""
    goal: str = ""
    plan_json: dict = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)      # 执行的步骤摘要
    failures: list[str] = Field(default_factory=list)     # 失败/错误
    outcome: str = ""                                      # 最终结果
    success: bool = False
    duration_seconds: int = 0
    tenant_id: int = 0
    user_id: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Lesson(BaseModel):
    """A learned lesson extracted from an Episode."""

    lesson_id: str = ""
    episode_id: str = ""
    lesson_type: LessonType = LessonType.INSIGHT
    content: str = ""                # 经验内容
    trigger_context: str = ""        # 什么情况下触发此经验
    action_advice: str = ""          # 具体行动建议
    success_rate: float = 0.0        # 此经验的成功率
    times_applied: int = 0
    importance: float = 0.5
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
```

### Task 2: Episodic Store + Lesson Extractor + Retriever + Ranker

**Files:**
- Create: `backend/memory/episodic_store.py`
- Create: `backend/memory/lesson_extractor.py`
- Create: `backend/memory/episode_retriever.py`
- Create: `backend/memory/memory_ranker.py`

```python
# backend/memory/episodic_store.py
"""EpisodicStore: persists Episodes and Lessons to Neo4j."""

class EpisodicStore:
    def save_episode(self, episode: Episode) -> bool:
        cypher = """
            MERGE (ep:Episode {episode_id: $eid})
            ON CREATE SET ep.goal = $goal, ep.outcome = $outcome,
                ep.success = $success, ep.duration_seconds = $dur,
                ep.tenant_id = $tid, ep.user_id = $uid, ep.created_at = $created
        """
        write_cypher(cypher, {...})
        return True

    def save_lesson(self, lesson: Lesson) -> bool:
        cypher = """
            MERGE (l:Lesson {lesson_id: $lid})
            ON CREATE SET l.content = $content, l.lesson_type = $type,
                l.trigger_context = $trigger, l.action_advice = $advice
            ON MATCH SET l.times_applied = l.times_applied + 1
        """
        # Also link: (l)-[:DERIVED_FROM]->(:Episode {episode_id: $eid})
        ...

    def get_similar_episodes(self, goal: str, user_id: int, tenant_id: int, limit: int = 5) -> list[Episode]:
        """Find past episodes with similar goals (keyword match for MVP)."""
        ...

    def get_lessons_by_type(self, lesson_type: LessonType, tenant_id: int, limit: int = 10) -> list[Lesson]:
        ...


# backend/memory/lesson_extractor.py
"""LessonExtractor: LLM extracts lessons from completed research."""

_LESSON_PROMPT = """从以下研究经验中提炼教训。输出JSON:
{
  "lessons": [
    {"lesson_type": "strategy|tool|pitfall|insight", "content": "...",
     "trigger_context": "...", "action_advice": "...", "importance": 0.8}
  ]
}"""

class LessonExtractor:
    async def extract(self, episode: Episode) -> list[Lesson]:
        """LLM analyzes an Episode and extracts actionable lessons."""
        model = init_chat_model("qwen-turbo", max_tokens=1024, ...)
        ...


# backend/memory/episode_retriever.py
"""EpisodeRetriever: retrieves relevant past episodes for new research."""

class EpisodeRetriever:
    def retrieve(self, goal: str, user_id: int, tenant_id: int) -> str:
        """Format relevant episodes as context string for Planner injection."""
        store = get_episodic_store()
        episodes = store.get_similar_episodes(goal, user_id, tenant_id, limit=3)
        if not episodes:
            return ""
        lines = ["## 历史研究经验"]
        for ep in episodes:
            status = "成功" if ep.success else "失败"
            lines.append(f"- [{status}] {ep.goal}: {ep.outcome[:100]}")
            if ep.failures:
                lines.append(f"  教训: {'; '.join(ep.failures[:2])}")
        return "\n".join(lines)


# backend/memory/memory_ranker.py
"""MemoryRanker: ranks historical experiences by relevance + recency + success."""

class MemoryRanker:
    def rank_episodes(self, episodes: list[Episode], current_goal: str) -> list[Episode]:
        """Score each episode: 0.3*similarity + 0.3*recency + 0.4*success_bonus."""
        ...

    def rank_lessons(self, lessons: list[Lesson], context: str) -> list[Lesson]:
        """Score lessons by relevance to current context."""
        ...
```

---

## Phase 2: Executor Integration

### Task 3: Hook into Research Executor

**Files:**
- Modify: `backend/research/executor.py`
- Modify: `backend/research/planner.py`

在 executor 的研究完成后（`_generate_report` 之后）调用:

```python
# v22: Save Episode and extract Lessons after research completes
from backend.memory.episodic_store import get_episodic_store
from backend.memory.lesson_extractor import get_lesson_extractor

episode = Episode(
    episode_id=f"ep_{uuid.uuid4().hex[:12]}",
    goal=plan.goal,
    actions=list(state.task_results.keys()),
    failures=[r.get("error", "") for r in state.task_results.values() if "error" in r],
    outcome="completed" if state.status == ResearchTaskStatus.COMPLETED else "failed",
    success=(state.status == ResearchTaskStatus.COMPLETED),
    duration_seconds=int((datetime.now(timezone.utc) - started).total_seconds()),
    tenant_id=tenant_id, user_id=user_id,
)
store = get_episodic_store()
store.save_episode(episode)

extractor = get_lesson_extractor()
lessons = await extractor.extract(episode)
for lesson in lessons:
    store.save_lesson(lesson)
```

在 planner 生成计划前注入历史经验:

```python
# v22: Inject relevant past episodes into planner context
from backend.memory.episode_retriever import get_episode_retriever
retriever = get_episode_retriever()
episode_context = retriever.retrieve(goal, user_id, tenant_id)
if episode_context:
    goal = f"{episode_context}\n\n当前目标: {goal}"
```

---

## Self-Review

| 610 文档 v22 需求 | 覆盖 |
|---|---|
| Episode Schema (goal/plan/actions/failures/outcome/lessons) | Task 1 (schemas.py) |
| Episodic Store (Neo4j 持久化) | Task 2 (episodic_store.py) |
| Lesson Extractor (LLM 提炼经验) | Task 2 (lesson_extractor.py) |
| Episode Retriever (检索相似经验) | Task 2 (episode_retriever.py) |
| Memory Ranker (时效+相似+成功权重) | Task 2 (memory_ranker.py) |
| Executor Hook (研究完成→保存Episode) | Task 3 (executor.py) |
| Planner Hook (新研究→注入经验) | Task 3 (planner.py) |
