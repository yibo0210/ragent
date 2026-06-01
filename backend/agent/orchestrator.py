"""多智能体 Supervisor 编排图模块

基于 LangGraph StateGraph 实现 Supervisor-Workers 架构：
- Supervisor Agent: 意图识别与路由分发
- RAG Specialist: 知识库深度检索（复用现有 rag_pipeline）
- Web Searcher: 联网实时搜索
- Direct Answer: 直接回答（闲聊、通用知识）

图拓扑：
    START -> supervisor -> [conditional] -> rag_specialist | web_searcher | direct_answer
    rag_specialist -> synthesize -> END
    web_searcher  -> synthesize -> END
    direct_answer -> END
"""
from typing import Annotated, Literal, Optional, TypedDict
import os
import time
import json
import re

from dotenv import load_dotenv
from backend.observability import get_tracer, get_logger, Metrics

tracer = get_tracer("ragent.agent")
log = get_logger("ragent.agent")
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------
API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", MODEL)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))

# 模型单例（延迟初始化）
_supervisor_model = None
_worker_model = None


def _get_supervisor_model():
    """获取 Supervisor 路由决策模型（结构化输出）。"""
    global _supervisor_model
    if _supervisor_model is None:
        from backend.agent.model_router import get_model_for_agent
        _supervisor_model = get_model_for_agent("supervisor")
    return _supervisor_model


def _get_worker_model():
    """获取 Worker 回答生成模型。"""
    global _worker_model
    if _worker_model is None:
        from backend.agent.model_router import get_model_for_agent
        _worker_model = get_model_for_agent("rag_specialist")
    return _worker_model


# ---------------------------------------------------------------------------
# 结构化输出 Schema
# ---------------------------------------------------------------------------
class SupervisorRoute(BaseModel):
    """Supervisor 路由决策结果。routes 可包含 1~3 个 Worker 实现并行调度。"""
    routes: list[str] = Field(
        description="路由目标列表，可包含: rag_specialist, web_searcher, direct_answer, data_analyst, local_graph_search, global_graph_search"
    )
    reason: str = Field(
        description="路由原因，简要说明为什么选择这些 Worker"
    )
    is_temporal: bool = Field(default=False, description="是否为时序敏感查询")
    temporal_year: str = Field(default="", description="时序查询目标年份")


# ---------------------------------------------------------------------------
# 图状态定义
# ---------------------------------------------------------------------------
class SupervisorState(TypedDict):
    """多智能体全局状态。"""
    # 对话消息历史（自动合并）
    messages: Annotated[list[BaseMessage], add_messages]
    # Supervisor 路由决策（单路由，向后兼容）
    next_worker: str
    # 并行路由目标列表
    next_workers: list[str]
    # 路由原因
    route_reason: str
    # 用户原始问题（方便 Worker 节点使用）
    user_query: str
    # RAG 检索 trace（从 RAG Specialist 透传）
    rag_trace: Optional[dict]
    # 联网搜索 trace
    web_search_trace: Optional[dict]
    # Agent 路由 trace（用于前端展示和持久化）
    agent_trace: Optional[dict]
    # Worker 各自汇报的数据片段
    worker_outputs: dict
    # 人类介入时的输入指令或修正文本
    human_interfered_input: str
    # v8: 自适应推理状态
    query_plan: Optional[dict]          # Planner 生成的分步计划
    critique_result: Optional[dict]     # Critique 评估结果
    retry_count: int                    # 当前重试次数（防止死循环）
    draft_answer: str                   # Synthesize 生成的草稿答案
    is_hallucinated: bool               # 幻觉标记
    plan_steps_completed: list[str]     # 已完成的计划步骤
    # v9: MCP 工具状态
    tool_outputs: dict                  # {step_id: output_data} MCP 调用结果
    # v12: Query 意图画像
    query_intent: Optional[dict]        # Profiler 输出的意图标签


# ---------------------------------------------------------------------------
# Supervisor 路由提示词
# ---------------------------------------------------------------------------
SUPERVISOR_SYSTEM_PROMPT = """你是一个智能路由调度员。你的职责是分析用户问题，将其路由到最合适的 Worker。

## 路由规则

### rag_specialist（知识库专家）
当用户问题涉及以下内容时路由至此：
- 上传的文档、内部知识库内容
- 要求总结、搜索、解释已有文档
- 与知识库中已有材料相关的深度问题

### local_graph_search（图谱局部检索）
当用户问题涉及以下内容时路由至此：
- 实体间的关系查询（"A和B什么关系？"）
- 多跳推理（"某个技术依赖哪些组件？"）
- 需要关联多个文档片段的复杂问题

### global_graph_search（图谱全局检索）
当用户问题涉及以下内容时路由至此：
- 总结性提问（"有哪些主要技术？"）
- 全局对比（"所有论文的方法有什么区别"）
- 全景分析（"整体技术架构是怎样的"）

### web_searcher（联网搜索专家）
当用户问题涉及以下内容时路由至此：
- 天气查询、实时新闻、最新动态、股价、实时信息（必须联网搜索）
- 需要联网获取的实时信息（股价、赛事、天气预报等）
- 知识库不太可能覆盖的通用常识或最新信息

### direct_answer（直接回答）
当用户问题属于以下类型时路由至此：
- 简单问候、闲聊
- 通用知识问答（不需要检索特定文档或联网）
- 天气查询等工具类请求
- 关于系统本身的提问

### data_analyst（数据分析师）
当用户问题涉及以下内容时路由至此：
- 统计数据、计数、排行（如"有多少条会话记录"）
- 结构化数据查询（如"最近的消息有哪些"）
- 数据汇总、聚合分析

### multimodal_specialist（多模态专家）
当用户问题涉及视觉/图表内容时路由至此：
- 提到"图表"、"曲线"、"图片"、"趋势图"、"柱状图"、"饼图"、"示意图"
- 涉及图像解读、图表分析等需求

## 时序路由规则
如果用户问题涉及时间跨度（如"去年"、"2023年到2025年"、"最近"、"最新的"、"X年的"），设置 is_temporal=true 并提取 temporal_year，优先路由到 local_graph_search。

## 注意事项
- 实时信息（天气、新闻、股价）必须选 web_searcher，不要选 direct_answer
- 大多数情况选择一个路由即可
- 当问题需要知识库+联网信息互补时，可同时选择多个（如 rag_specialist + web_searcher）
- 不要同时选择 direct_answer 和其他 Worker
- 简要说明选择原因（10-30字）"""


# ---------------------------------------------------------------------------
# Worker 系统提示词
# ---------------------------------------------------------------------------
RAG_SPECIALIST_PROMPT = """你是一个知识库专家。根据检索到的文档片段回答用户问题。
- 基于提供的上下文回答，不要编造信息
- 如果上下文不足以回答，诚实说明
- 保持回答简洁、条理清晰
- 使用中文回答"""

WEB_SEARCHER_PROMPT = """你是一个联网搜索专家。根据搜索结果回答用户问题。
- 基于搜索结果回答，引用来源
- 如果搜索结果不足以回答，诚实说明
- 保持回答简洁、准确
- 使用中文回答"""

DIRECT_ANSWER_PROMPT = """你是一个 AI 学习助手，帮助用户学习和掌握 AI 的使用方法。
- 通俗易懂、条理清晰
- 适合不同水平的用户理解
- 保持友好、耐心、专业的语气
- 使用中文回答"""

DATA_ANALYST_PROMPT = """你是一个数据分析师。根据 SQL 查询结果回答用户问题。
- 基于查询结果回答，解读数据含义
- 如果查询失败或结果为空，诚实说明
- 提供数据洞察，而不仅仅是罗列数字
- 使用中文回答"""


# ---------------------------------------------------------------------------
# 流式回答辅助函数
# ---------------------------------------------------------------------------
def _stream_answer(model, messages: list) -> str:
    """流式调用 LLM 生成回答，逐 token 推送到前端。失败时降级为非流式。"""
    import traceback
    from .tools import emit_token

    full_answer = ""
    try:
        for chunk in model.stream(messages):
            token = chunk.content if hasattr(chunk, "content") else ""
            if token:
                full_answer += token
                emit_token(token)
    except Exception:
        print(f"[WARN] stream() 失败，降级到 invoke(): {traceback.format_exc()}")
        try:
            response = model.invoke(messages)
            full_answer = response.content if hasattr(response, "content") else str(response)
            if full_answer:
                emit_token(full_answer)
        except Exception:
            print(f"[ERROR] invoke() 也失败: {traceback.format_exc()}")
            full_answer = "抱歉，生成回答时出现了错误，请稍后重试。"
            emit_token(full_answer)
    return full_answer


# ---------------------------------------------------------------------------
# 图节点实现
# ---------------------------------------------------------------------------
def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor 节点：分析用户意图，决定路由方向。"""
    messages = state["messages"]

    # 提取用户最新问题
    user_query = state.get("user_query", "")
    if not user_query:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

    # v12: Query Profiler — 轻量级意图分类
    query_intent = None
    try:
        from backend.agent.query_profiler import QueryProfiler
        profiler = QueryProfiler(use_embedding=True)
        intent = profiler.profile(user_query)
        query_intent = intent.to_dict()
        log.info("query_profiled", level=intent.level, score=intent.complexity_score,
                 keywords=intent.matched_keywords[:3])
    except Exception:
        query_intent = None

    # v12: L1 简单查询快速通道 — 跳过 Supervisor LLM，直接路由到 direct_answer
    if query_intent and query_intent["level"] == "L1_FACTUAL" and query_intent["complexity_score"] < 0.3:
        agent_trace = state.get("agent_trace") or {}
        agent_trace["routing_agent"] = "direct_answer"
        agent_trace["routing_agents"] = ["direct_answer"]
        agent_trace["routing_reason"] = f"Query Profiler 快速通道: L1 事实类 (score={query_intent['complexity_score']})"
        agent_trace["routing_is_temporal"] = False
        agent_trace["routing_temporal_year"] = ""
        agent_trace["query_intent"] = query_intent
        log.info("fast_route_l1", query=user_query[:50])
        Metrics.record_routing("direct_answer")
        return {
            "next_worker": "direct_answer",
            "next_workers": ["direct_answer"],
            "route_reason": agent_trace["routing_reason"],
            "user_query": user_query,
            "agent_trace": agent_trace,
            "worker_outputs": {},
            "human_interfered_input": "",
            "query_intent": query_intent,
        }

    # 调用 Supervisor LLM 进行路由决策（手动 JSON 解析）
    model = _get_supervisor_model()
    route_prompt = SUPERVISOR_SYSTEM_PROMPT + (
        '\n\n请严格输出JSON格式，不要包含其他内容：\n'
        '{"routes": ["agent_name"], "reason": "选择原因"}\n'
        f'\n用户问题：{user_query}'
    )

    try:
        response = model.invoke([HumanMessage(content=route_prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r'\{[^{}]*"routes"\s*:\s*\[[^\]]*\][^{}]*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            routes = data.get("routes", ["rag_specialist"])
            reason = data.get("reason", "")
            is_temporal = data.get("is_temporal", False)
            temporal_year = data.get("temporal_year", "")
        else:
            content_lower = content.lower()
            if "web_search" in content_lower or "联网" in content_lower:
                routes = ["web_searcher"]; reason = "联网搜索"
            else:
                routes = ["rag_specialist"]; reason = "路由解析失败"
            is_temporal = False; temporal_year = ""
        if isinstance(routes, str):
            routes = [routes]
    except Exception:
        routes = ["rag_specialist"]
        reason = "路由解析失败，默认知识库检索"
        is_temporal = False
        temporal_year = ""

    # 过滤无效路由，去重
    valid = {"rag_specialist", "web_searcher", "direct_answer", "data_analyst", "local_graph_search", "global_graph_search", "multimodal_specialist", "planner"}
    routes = [r for r in routes if r in valid]
    if not routes:
        routes = ["direct_answer"]

    # 构建 agent_trace
    agent_trace = state.get("agent_trace") or {}
    agent_trace["routing_agent"] = routes[0]
    agent_trace["routing_agents"] = routes
    agent_trace["routing_reason"] = reason
    agent_trace["routing_is_temporal"] = is_temporal
    agent_trace["routing_temporal_year"] = temporal_year
    agent_trace["query_intent"] = query_intent

    log.info("routing_decision", routes=routes, reason=reason, query=user_query[:100])
    for r in routes:
        Metrics.record_routing(r)

    return {
        "next_worker": routes[0],
        "next_workers": routes,
        "route_reason": reason,
        "user_query": user_query,
        "agent_trace": agent_trace,
        "worker_outputs": {},
        "human_interfered_input": "",
        "query_intent": query_intent,
    }


def rag_specialist_node(state: SupervisorState) -> dict:
    """RAG Specialist 节点：执行知识库检索并生成回答。"""
    from backend.rag.pipeline import run_rag_graph
    from .tools import emit_rag_step

    user_query = state.get("user_query", "")
    if not user_query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

    # v12: 传递意图标签到 RAG pipeline
    query_intent = state.get("query_intent")
    intent_level = query_intent.get("level") if query_intent else None

    # 执行 RAG 检索流水线
    rag_result = run_rag_graph(user_query, intent_level=intent_level)
    docs = rag_result.get("docs", [])
    context = rag_result.get("context", "")
    rag_trace = rag_result.get("rag_trace", {})

    # Scenario A: RAG 低置信度 → HITL 中断
    if rag_result.get("force_interrupt"):
        emit_rag_step("🛑", "HITL 中断 — RAG检索置信度过低，挂起工作流等待人工决策")
        interrupt({
            "type": "hitl_rag_grade",
            "scenario": "low_confidence_rag",
            "query": user_query,
            "grade_score": rag_trace.get("grade_score_v2", rag_trace.get("grade_score")),
            "grade_fail_count": rag_trace.get("grade_fail_count", 2),
            "docs": docs,
            "message": "知识库检索两次评分均未通过，请审核并提供指导。",
        })

    # 用 Worker LLM 基于检索结果生成回答
    model = _get_worker_model()

    if context:
        prompt = f"{RAG_SPECIALIST_PROMPT}\n\n## 检索到的文档\n\n{context}\n\n## 用户问题\n\n{user_query}"
    else:
        prompt = f"{RAG_SPECIALIST_PROMPT}\n\n未检索到相关文档，请根据你的知识回答：\n\n{user_query}"

    answer = _stream_answer(model, [HumanMessage(content=prompt)])

    # 更新 agent_trace
    agent_trace = state.get("agent_trace") or {}
    agent_trace["rag_trace"] = rag_trace

    worker_outputs = state.get("worker_outputs") or {}
    worker_outputs["rag_specialist"] = {"answer": answer, "trace": rag_trace}

    return {
        "messages": [AIMessage(content=answer)],
        "rag_trace": rag_trace,
        "agent_trace": agent_trace,
        "worker_outputs": worker_outputs,
    }


def web_searcher_node(state: SupervisorState) -> dict:
    """Web Searcher 节点：执行联网搜索并生成回答。"""
    from .web_searcher import run_web_search, format_web_search_context

    user_query = state.get("user_query", "")
    if not user_query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

    # v12: CRITICAL 状态下跳过 Tavily 搜索
    from backend.ha.load_monitor import get_load_monitor
    monitor = get_load_monitor()
    if monitor.should_circuit_break_tavily():
        search_result = {"error": "CRITICAL 负载降级，跳过联网搜索", "results": []}
    else:
        # 执行联网搜索
        search_result = run_web_search(user_query)

    # 搜索失败时降级到 RAG Specialist
    if search_result.get("error") or not search_result.get("results"):
        from backend.rag.pipeline import run_rag_graph

        rag_result = run_rag_graph(user_query)
        docs = rag_result.get("docs", [])
        context_rag = rag_result.get("context", "")
        rag_trace = rag_result.get("rag_trace", {})

        fallback_reason = search_result.get("error") or "搜索结果为空"
        model = _get_worker_model()
        if context_rag:
            prompt = f"{RAG_SPECIALIST_PROMPT}\n\n## 检索到的文档\n\n{context_rag}\n\n## 用户问题\n\n{user_query}"
        else:
            prompt = f"{RAG_SPECIALIST_PROMPT}\n\n未检索到相关文档，请根据你的知识回答：\n\n{user_query}"
        answer = _stream_answer(model, [HumanMessage(content=prompt)])

        agent_trace = state.get("agent_trace") or {}
        agent_trace["fallback"] = {
            "original_worker": "web_searcher",
            "fallback_worker": "rag_specialist",
            "reason": fallback_reason,
        }
        agent_trace["rag_trace"] = rag_trace

        worker_outputs = state.get("worker_outputs") or {}
        worker_outputs["web_searcher"] = {"answer": answer, "trace": rag_trace, "fallback": True}

        return {
            "messages": [AIMessage(content=answer)],
            "rag_trace": rag_trace,
            "agent_trace": agent_trace,
            "worker_outputs": worker_outputs,
        }

    context = format_web_search_context(search_result)

    # 用 Worker LLM 基于搜索结果生成回答
    model = _get_worker_model()
    prompt = f"{WEB_SEARCHER_PROMPT}\n\n## 搜索结果\n\n{context}\n\n## 用户问题\n\n{user_query}"
    answer = _stream_answer(model, [HumanMessage(content=prompt)])

    # 构建 web_search_trace
    web_search_trace = {
        "tool_used": True,
        "tool_name": "search_web",
        "query": user_query,
        "results": search_result.get("results", []),
        "result_count": len(search_result.get("results", [])),
        "answer": search_result.get("answer", ""),
        "error": search_result.get("error"),
    }

    # 更新 agent_trace
    agent_trace = state.get("agent_trace") or {}
    agent_trace["web_search_trace"] = web_search_trace

    worker_outputs = state.get("worker_outputs") or {}
    worker_outputs["web_searcher"] = {"answer": answer, "trace": web_search_trace}

    return {
        "messages": [AIMessage(content=answer)],
        "web_search_trace": web_search_trace,
        "agent_trace": agent_trace,
        "worker_outputs": worker_outputs,
    }


def data_analyst_node(state: SupervisorState) -> dict:
    """Data Analyst 节点：Text-to-SQL 查询并生成回答。v9: 支持 MCP 外部数据源。"""
    from .data_analyst import get_schema_info, generate_sql, execute_sql, format_sql_result
    from .data_analyst import get_mcp_data_sources, generate_mcp_query, execute_mcp_query, format_mcp_result
    from .tools import emit_rag_step
    from .mcp_client import get_mcp_manager

    user_query = state.get("user_query", "")
    if not user_query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

    # v9: 检查是否有 MCP 数据源可用
    import asyncio
    mcp_sources = []
    try:
        loop = asyncio.get_event_loop()
        mcp_sources = loop.run_until_complete(get_mcp_data_sources())
    except Exception:
        pass

    # 本地 MySQL 查询
    emit_rag_step("📊", "Schema 发现 — Data Analyst 读取MySQL数据库表结构", agent="data_analyst")
    schema = get_schema_info()

    emit_rag_step("🔍", "Text-to-SQL — LLM 将自然语言转为SQL查询语句", agent="data_analyst")
    sql = generate_sql(user_query, schema)
    emit_rag_step("📝", f"生成SQL — {sql[:100]}", agent="data_analyst")

    result = execute_sql(sql)

    if result.get("error") == "non_select":
        emit_rag_step("🛑", "HITL 审核 — 检测到非SELECT语句，需人工批准后执行", agent="data_analyst")
        interrupt({
            "type": "hitl_sql_approval",
            "scenario": "non_select_sql",
            "sql": sql,
            "query": user_query,
            "message": "数据分析师生成了非 SELECT SQL 语句，请审核是否批准执行。",
        })

    emit_rag_step("✅", f"SQL执行完成 — 返回 {result.get('row_count', 0)} 行数据", agent="data_analyst")
    local_context = format_sql_result(result)

    # v9: MCP 数据源查询
    mcp_results = []
    if mcp_sources:
        emit_rag_step("🔗", f"发现 {len(mcp_sources)} 个 MCP 数据源，正在查询...", agent="data_analyst")
        for source in mcp_sources[:3]:  # 最多查 3 个
            try:
                tool_schema = {
                    "tool_name": source["tool_name"],
                    "description": source["description"],
                    "input_schema": source.get("input_schema", {}),
                }
                args = loop.run_until_complete(generate_mcp_query(user_query, tool_schema))
                if args:
                    emit_rag_step("📡", f"调用 {source['server_name']}/{source['tool_name']}", agent="data_analyst")
                    mcp_result = loop.run_until_complete(
                        execute_mcp_query(source["server_name"], source["tool_name"], args)
                    )
                    mcp_results.append({
                        "server": source["server_name"],
                        "tool": source["tool_name"],
                        "result": mcp_result,
                    })
            except Exception as e:
                mcp_results.append({
                    "server": source["server_name"],
                    "tool": source["tool_name"],
                    "result": {"error": str(e)},
                })

    # 组合上下文
    context = f"## 本地 MySQL 查询结果\n\n{local_context}"
    if mcp_results:
        context += "\n\n## MCP 外部数据源查询结果\n"
        for mr in mcp_results:
            formatted = format_mcp_result(mr["result"])
            context += f"\n### {mr['server']}/{mr['tool']}\n{formatted}\n"

    model = _get_worker_model()
    prompt = f"{DATA_ANALYST_PROMPT}\n\n{context}\n\n## 用户问题\n\n{user_query}"
    answer = _stream_answer(model, [HumanMessage(content=prompt)])

    data_trace = {
        "tool_used": True,
        "tool_name": "data_analyst",
        "sql": sql,
        "result": result,
        "mcp_sources": len(mcp_sources),
        "mcp_results": [{"server": mr["server"], "tool": mr["tool"]} for mr in mcp_results],
    }
    agent_trace = state.get("agent_trace") or {}
    agent_trace["data_analyst_trace"] = data_trace

    worker_outputs = state.get("worker_outputs") or {}
    worker_outputs["data_analyst"] = {"answer": answer, "trace": data_trace}

    return {
        "messages": [AIMessage(content=answer)],
        "agent_trace": agent_trace,
        "worker_outputs": worker_outputs,
    }


def direct_answer_node(state: SupervisorState) -> dict:
    """Direct Answer 节点：直接生成回答（闲聊、通用知识）。"""
    from backend.agent.model_router import get_model_for_agent
    model = get_model_for_agent("direct_answer")  # qwen-turbo，更快

    # 构建对话历史（保留最近 10 条）
    recent_messages = state["messages"][-10:]
    prompt_messages = [SystemMessage(content=DIRECT_ANSWER_PROMPT)] + list(recent_messages)

    answer = _stream_answer(model, prompt_messages)

    # 更新 agent_trace（透传 supervisor 的路由信息）
    agent_trace = state.get("agent_trace") or {}

    worker_outputs = state.get("worker_outputs") or {}
    worker_outputs["direct_answer"] = {"answer": answer}

    return {
        "messages": [AIMessage(content=answer)],
        "agent_trace": agent_trace,
        "worker_outputs": worker_outputs,
    }


def local_graph_search_node(state: SupervisorState) -> dict:
    """图谱局部检索节点：向量检索 + Neo4j 外扩（含降级保护）。"""
    from backend.rag.graph_retriever import local_graph_search
    from backend.ha.degradation import safe_graph_search
    from .tools import emit_graph_step

    user_query = state.get("user_query", "")
    if not user_query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

    agent_trace = state.get("agent_trace") or {}

    time_filter = None
    if agent_trace.get("routing_is_temporal") and agent_trace.get("routing_temporal_year"):
        time_filter = {
            "year": agent_trace["routing_temporal_year"],
            "mode": "at_or_after",
        }

    # v12: CRITICAL 状态下跳过图谱搜索，降级到纯向量
    from backend.ha.load_monitor import get_load_monitor
    monitor = get_load_monitor()
    if monitor.should_circuit_break_neo4j():
        emit_graph_step("⚡", "负载降级 — CRITICAL 状态，跳过 Neo4j 图谱搜索", agent="local_graph_search")
        from backend.rag.utils import retrieve_documents
        result = retrieve_documents(user_query, top_k=5)
        result["mode"] = "degraded_load_critical"
    else:
        emit_graph_step("🔍", "局部图谱检索 — Milvus向量定位实体 → Neo4j 1-hop外扩邻居", agent="local_graph_search")
        with tracer.start_as_current_span("agent.local_graph_search") as span:
            span.set_attribute("query", user_query[:200])
            result = safe_graph_search(user_query)

    emit_graph_step(
        "🔗",
        f"图谱外扩完成 — Neo4j返回 {len(result.get('graph_triples', []))} 条实体关系三元组",
        agent="local_graph_search",
    )

    model = _get_worker_model()
    prompt = f"根据以下信息回答问题。如果涉及实体间的关系，请明确指出关联。\n\n{result['context']}\n\n## 用户问题\n\n{user_query}"
    answer = _stream_answer(model, [HumanMessage(content=prompt)])

    agent_trace = state.get("agent_trace") or {}
    agent_trace["graph_search_mode"] = "local"
    agent_trace["graph_triples_count"] = len(result.get("graph_triples", []))

    worker_outputs = state.get("worker_outputs") or {}
    worker_outputs["local_graph_search"] = {"answer": answer, "triples_count": len(result.get("graph_triples", []))}

    return {
        "messages": [AIMessage(content=answer)],
        "agent_trace": agent_trace,
        "worker_outputs": worker_outputs,
    }


def global_graph_search_node(state: SupervisorState) -> dict:
    """图谱全局检索节点：社区摘要匹配。"""
    from backend.rag.graph_retriever import global_graph_search
    from .tools import emit_graph_step

    user_query = state.get("user_query", "")
    if not user_query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

    emit_graph_step("📊", "全局图谱检索 — Milvus社区摘要向量匹配（Leiden聚类生成）", agent="global_graph_search")
    result = global_graph_search(user_query)
    n_summaries = len(result.get("summaries", []))
    emit_graph_step("✅", f"摘要匹配完成 — 命中 {n_summaries} 个相关社区综述", agent="global_graph_search")

    model = _get_worker_model()
    prompt = f"根据以下社区摘要信息，给出一份综合分析回答。\n\n{result['context']}\n\n## 用户问题\n\n{user_query}"
    answer = _stream_answer(model, [HumanMessage(content=prompt)])

    agent_trace = state.get("agent_trace") or {}
    agent_trace["graph_search_mode"] = "global"
    agent_trace["community_count"] = n_summaries

    worker_outputs = state.get("worker_outputs") or {}
    worker_outputs["global_graph_search"] = {"answer": answer, "community_count": n_summaries}

    return {
        "messages": [AIMessage(content=answer)],
        "agent_trace": agent_trace,
        "worker_outputs": worker_outputs,
    }


def multimodal_specialist_node(state: SupervisorState) -> dict:
    """多模态专家节点：视觉检索 + 引用回答。"""
    from backend.agent.multimodal_specialist import multimodal_specialist_node as _impl
    return _impl(state)


def synthesize_node(state: SupervisorState) -> dict:
    """Synthesize 节点：汇总并行 Worker 结果，生成聚合回答。"""
    worker_outputs = state.get("worker_outputs", {})

    # 单 Worker 无需聚合，直接提取答案
    if len(worker_outputs) <= 1:
        for agent_name, output in worker_outputs.items():
            answer = output.get("answer", "") if isinstance(output, dict) else str(output)
            if answer:
                return {"draft_answer": answer}
        return {}

    # 多 Worker：LLM 聚合
    model = _get_worker_model()
    user_query = state.get("user_query", "")

    parts = []
    for agent_name, output in worker_outputs.items():
        answer = output.get("answer", "") if isinstance(output, dict) else str(output)
        if answer:
            label = {
                "rag_specialist": "知识库检索",
                "web_searcher": "联网搜索",
                "data_analyst": "数据分析",
                "local_graph_search": "图谱局部检索",
                "global_graph_search": "图谱全局检索",
            }.get(agent_name, agent_name)
            parts.append(f"### {label}\n{answer}")

    if not parts:
        return {}

    synthesis_prompt = f"""你是一个信息整合专家。以下是多个智能体对同一问题的不同回答，请将它们整合为一个条理清晰、内容完整的回答。

规则：
- 融合各来源信息，互补而非重复
- 如果各来源有矛盾，指出差异并给出综合判断
- 保持回答简洁、有条理
- 使用中文回答

## 各智能体回答

{chr(10).join(parts)}

## 用户原始问题

{user_query}"""

    answer = _stream_answer(model, [HumanMessage(content=synthesis_prompt)])

    return {"messages": [AIMessage(content=answer)], "draft_answer": answer}


# ---------------------------------------------------------------------------
# v8: Planner / Critique / Replan 节点
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """你是一个任务规划专家。分析用户问题，判断是否需要多步执行。

规则：
- 简单问题（单次检索可回答）：返回 is_complex=false
- 复杂问题（需要多跳推理、跨文档关联、多来源综合）：拆解为 2-4 个步骤
- 每个步骤指定 agent 和子查询
- 可用 agent: rag_specialist, local_graph_search, global_graph_search, web_searcher, data_analyst
- v9: 步骤支持 DAG 依赖关系（dependencies）和数据传递（input_mapping）
- 如果有 critique_feedback（之前回答被驳回），据此调整计划补充缺失信息

输出严格JSON格式：
{"is_complex": bool, "steps": [{"step_id": 1, "agent": "...", "query": "...", "tool_name": "", "dependencies": [], "input_mapping": {}}], "reasoning": "..."}"""

CRITIQUE_PROMPT = """你是一个严格的事实核查专家。检查以下回答是否完全基于提供的上下文。

规则：
- 逐条检查回答中的事实声明
- 每个声明必须能在上下文中找到直接依据
- 如果有声明无法验证或明显编造，标记 is_valid=false
- 提供 missing_information 列表说明缺失哪些关键信息

上下文：
{contexts}

草稿回答：
{draft_answer}

用户问题：
{user_query}

输出严格JSON格式：
{{"is_valid": bool, "missing_information": ["..."], "feedback": "...", "confidence": 0.0-1.0}}"""


def planner_node(state: SupervisorState) -> dict:
    """Planner 节点：将复杂查询拆解为多步执行计划。"""
    from .model_router import get_model_for_agent

    user_query = state.get("user_query", "")
    if not user_query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

    critique_feedback = ""
    critique_result = state.get("critique_result")
    if critique_result and not critique_result.get("is_valid", True):
        critique_feedback = f"\n\n之前的回答被驳回，原因：{critique_result.get('feedback', '')}\n缺失信息：{', '.join(critique_result.get('missing_information', []))}"

    model = get_model_for_agent("supervisor")
    prompt = PLANNER_PROMPT + f"\n\n用户问题：{user_query}{critique_feedback}"

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r'\{[^{}]*"steps"\s*:\s*\[[^\]]*\][^{}]*\}', content, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group())
        else:
            plan = {"is_complex": False, "steps": [], "reasoning": "解析失败，降级为简单路由"}
    except Exception:
        plan = {"is_complex": False, "steps": [], "reasoning": "Planner 调用失败"}

    agent_trace = state.get("agent_trace") or {}
    agent_trace["plan"] = plan

    log.info("planner_decision", is_complex=plan.get("is_complex"), steps=len(plan.get("steps", [])))

    return {
        "query_plan": plan,
        "agent_trace": agent_trace,
    }


def critique_node(state: SupervisorState) -> dict:
    """Critique 节点：交叉验证草稿回答与检索依据。"""
    from .model_router import get_model_for_agent

    # 提取草稿答案
    draft = state.get("draft_answer", "")
    if not draft:
        result_messages = state.get("messages", [])
        if result_messages:
            last_msg = result_messages[-1]
            draft = getattr(last_msg, "content", "")

    if not draft:
        return {"critique_result": {"is_valid": True, "feedback": "无草稿答案", "confidence": 1.0}}

    # 提取上下文
    contexts = []
    worker_outputs = state.get("worker_outputs", {})
    for agent_name, output in worker_outputs.items():
        if isinstance(output, dict):
            trace = output.get("trace", {})
            if isinstance(trace, dict):
                chunks = trace.get("retrieved_chunks", [])
                if isinstance(chunks, list):
                    for chunk in chunks:
                        if isinstance(chunk, dict) and chunk.get("text"):
                            contexts.append(chunk["text"])

    user_query = state.get("user_query", "")
    context_text = "\n\n---\n\n".join(contexts[:10]) if contexts else "未检索到上下文"

    model = get_model_for_agent("supervisor")
    prompt = CRITIQUE_PROMPT.format(
        contexts=context_text[:3000],
        draft_answer=draft[:2000],
        user_query=user_query,
    )

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r'\{[^{}]*"is_valid"\s*:\s*(?:true|false)[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"is_valid": True, "feedback": "Critique 解析失败，默认通过", "confidence": 0.5}
    except Exception:
        result = {"is_valid": True, "feedback": "Critique 调用失败", "confidence": 0.5}

    retry = state.get("retry_count", 0)
    log.info("critique_result", is_valid=result.get("is_valid"), retry=retry)

    return {
        "critique_result": result,
        "is_hallucinated": not result.get("is_valid", True),
    }


def replan_node(state: SupervisorState) -> dict:
    """Replan 节点：根据 Critique 反馈重新规划查询。"""
    critique = state.get("critique_result", {})
    missing = critique.get("missing_information", [])
    feedback = critique.get("feedback", "")
    retry = state.get("retry_count", 0) + 1

    # 构建补充查询
    original_query = state.get("user_query", "")
    supplement = f"补充查询：{original_query}\n需要补充的信息：{', '.join(missing)}"
    if feedback:
        supplement += f"\n反馈：{feedback}"

    return {
        "user_query": supplement,
        "retry_count": retry,
        "worker_outputs": {},
        "draft_answer": "",
        "query_plan": None,
        "critique_result": None,
        "is_hallucinated": False,
    }


def route_after_critique(state: SupervisorState) -> str:
    """Critique 后的条件路由。"""
    from backend.ha.load_monitor import get_load_monitor
    critique = state.get("critique_result", {})
    retry = state.get("retry_count", 0)

    # v12: 高负载时跳过 Critique 重试，直接输出
    monitor = get_load_monitor()
    if monitor.should_skip_critique() and not critique.get("is_valid", True):
        log.info("critique_skipped_due_to_load", state=monitor.get_state().value)
        return "end"

    if critique.get("is_valid", True):
        return "end"
    elif retry < 2:
        return "replan"
    else:
        return "end"


# ---------------------------------------------------------------------------
# 条件路由函数
# ---------------------------------------------------------------------------
def route_supervisor(state: SupervisorState):
    """根据 Supervisor 决策返回目标节点。支持 Send 并行调度。"""
    # v8: 如果有 query_plan（来自 planner），按计划步骤路由
    plan = state.get("query_plan")
    if plan and plan.get("is_complex") and plan.get("steps"):
        valid_agents = {"rag_specialist", "web_searcher", "data_analyst",
                        "local_graph_search", "global_graph_search", "multimodal_specialist"}
        sends = []
        for step in plan["steps"]:
            agent = step.get("agent", "rag_specialist")
            if agent in valid_agents:
                step_state = {**state, "user_query": step.get("query", state.get("user_query", ""))}
                sends.append(Send(agent, step_state))
        if sends:
            return sends

    # 原有逻辑
    workers = state.get("next_workers", [])
    if not workers:
        workers = [state.get("next_worker", "direct_answer")]

    # 单个 direct_answer 直接返回（不需要 synthesize）
    if workers == ["direct_answer"]:
        return "direct_answer"

    # v8: 如果路由到 planner，直接返回
    if workers == ["planner"]:
        return "planner"

    # 其他情况用 Send fan-out 到各 Worker
    sends = []
    for worker in workers:
        if worker == "direct_answer":
            sends.append(Send("direct_answer", state))
        elif worker == "planner":
            return "planner"
        else:
            sends.append(Send(worker, state))
    return sends


# ---------------------------------------------------------------------------
# 构建 Supervisor 图
# ---------------------------------------------------------------------------
def build_supervisor_graph(checkpointer=None):
    """构建并编译 Supervisor LangGraph 状态机（v8: 含 Planner/Critique/Replan）。"""
    graph = StateGraph(SupervisorState)

    # 节点注册
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("rag_specialist", rag_specialist_node)
    graph.add_node("web_searcher", web_searcher_node)
    graph.add_node("data_analyst", data_analyst_node)
    graph.add_node("local_graph_search", local_graph_search_node)
    graph.add_node("global_graph_search", global_graph_search_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("multimodal_specialist", multimodal_specialist_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("critique", critique_node)
    graph.add_node("replan", replan_node)

    # 入口
    graph.set_entry_point("supervisor")

    # Supervisor 条件路由（支持 planner 和 Send fan-out）
    graph.add_conditional_edges("supervisor", route_supervisor)

    # Worker -> Synthesize（数据分析师和直接回答跳过 Critique）
    for worker in ["rag_specialist", "web_searcher",
                   "local_graph_search", "global_graph_search", "multimodal_specialist"]:
        graph.add_edge(worker, "synthesize")

    # Synthesize -> Critique（v8: 不再直接到 END）
    graph.add_edge("synthesize", "critique")

    # Direct Answer -> END（闲聊无需 Critique 校验）
    graph.add_edge("direct_answer", END)

    # Data Analyst -> END（SQL 查询结果是结构化数据，非 RAG 检索上下文，Critique 会误判）
    graph.add_edge("data_analyst", END)

    # Critique 条件路由
    graph.add_conditional_edges("critique", route_after_critique, {
        "end": END,
        "replan": "replan",
    })

    # Replan -> Supervisor（重新路由）
    graph.add_edge("replan", "supervisor")

    return graph.compile(checkpointer=checkpointer)


_cached_graph = None


def _get_supervisor_graph():
    """获取带 checkpointer 的图单例（延迟初始化 + 缓存）。"""
    global _cached_graph
    if _cached_graph is None:
        from backend.storage.checkpointer import _get_checkpointer
        _cached_graph = build_supervisor_graph(checkpointer=_get_checkpointer())
    return _cached_graph


# 向后兼容：无 checkpointer 的模块级单例
supervisor_graph = build_supervisor_graph()
