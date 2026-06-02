"""AI 工具集模块

提供以下能力：
- 天气查询（get_current_weather）
- 知识库检索（search_knowledge_base）
- 流式步骤推送（emit_rag_step）
- 工具调用限制（防止重复查库）
"""
from typing import Optional
import os
import requests
from dotenv import load_dotenv
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool

load_dotenv()

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

_LAST_RAG_CONTEXT = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_RAG_STEP_QUEUE = None  # asyncio.Queue, set by agent before streaming
_RAG_STEP_LOOP = None   # asyncio loop, captured when setting queue

_TOKEN_QUEUE = None  # asyncio.Queue proxy, for token-level streaming
_TOKEN_LOOP = None   # asyncio loop, captured when setting queue


def _set_last_rag_context(context: dict):
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    if queue:
        import asyncio
        try:
            _RAG_STEP_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            _RAG_STEP_LOOP = asyncio.get_event_loop()
    else:
        _RAG_STEP_LOOP = None


def emit_rag_step(icon: str, label: str, detail: str = "", agent: str = "rag_specialist"):
    """向队列发送一个检索步骤。支持跨线程安全调用。

    Args:
        icon: 步骤图标 emoji
        label: 步骤标签
        detail: 步骤详情
        agent: 来源 Agent 标识（rag_specialist / web_searcher）
    """
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
        step = {"icon": icon, "label": label, "detail": detail, "agent": agent}
        try:
            if not _RAG_STEP_LOOP.is_closed():
                _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
        except Exception:
            pass  # SSE queue closed, caller disconnected


def emit_graph_step(icon: str, message: str, agent: str = "local_graph_search"):
    """向队列发送图谱检索步骤。"""
    emit_rag_step(icon, message, detail="", agent=agent)


def set_token_queue(queue):
    """设置 token 流式传输队列，供 worker 节点推送流式 token。"""
    global _TOKEN_QUEUE, _TOKEN_LOOP
    _TOKEN_QUEUE = queue
    if queue:
        import asyncio
        try:
            _TOKEN_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            _TOKEN_LOOP = asyncio.get_event_loop()
    else:
        _TOKEN_LOOP = None


def emit_token(token: str):
    """向队列发送一个流式 token。支持跨线程安全调用。"""
    global _TOKEN_QUEUE, _TOKEN_LOOP
    if _TOKEN_QUEUE is not None and _TOKEN_LOOP is not None and token:
        try:
            if not _TOKEN_LOOP.is_closed():
                _TOKEN_LOOP.call_soon_threadsafe(
                    _TOKEN_QUEUE.put_nowait,
                    {"type": "content", "content": token}
                )
        except Exception:
            pass  # SSE queue closed, caller disconnected


def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """获取天气信息"""
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    # ... guards omitted ...
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1

    from backend.rag.pipeline import run_rag_graph

    # 在同步工具中获取当前的 Loop 可能不可靠，但我们之前是通过 call_soon_threadsafe 调度的。
    # 这里 _RAG_STEP_QUEUE 是在主线程/Loop 设置的全局变量。
    # 如果工具运行在线程池中，它是可以访问到全局变量 _RAG_STEP_QUEUE 的。
    # emit_rag_step 内部做了 try-except 和 get_event_loop()。

    # 问题可能出在 asyncio.get_event_loop() 在子线程中调用会报错或者拿不到主线程的loop。
    # 我们应该在 set_rag_step_queue 时也保存 loop 引用，或者在 emit_rag_step 中更健壮地获取 loop。

    rag_result = run_rag_graph(query)

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})

    if not docs:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)


@tool("search_web")
def search_web(query: str) -> str:
    """Search the web for real-time information, current events, or topics not in the knowledge base."""
    from .web_searcher import run_web_search, format_web_search_context

    result = run_web_search(query)
    context = format_web_search_context(result)

    if result.get("error"):
        return f"Web search failed: {result['error']}"

    return context if context else "No web search results found."


# ---------------------------------------------------------------------------
# v9: MCP 动态工具注册
# ---------------------------------------------------------------------------
def mcp_tools_to_langchain_tools(mcp_tools: list[dict], server_name: str) -> list:
    """将 MCP 工具 Schema 转换为 LangChain Tool 对象。

    Args:
        mcp_tools: MCP tools/list 返回的工具列表 [{name, description, input_schema}]
        server_name: MCP Server 名称，用于调用时路由

    Returns:
        LangChain Tool 对象列表
    """
    from langchain_core.tools import StructuredTool
    from pydantic import create_model, Field

    tools = []
    for tool_def in mcp_tools:
        name = tool_def.get("name", "")
        description = tool_def.get("description", "")
        input_schema = tool_def.get("input_schema", {})

        # 从 JSON Schema 构建 Pydantic 模型
        properties = input_schema.get("properties", {})
        required_fields = set(input_schema.get("required", []))
        field_defs = {}
        for param_name, param_def in properties.items():
            param_type = _json_type_to_python(param_def.get("type", "string"))
            default = ... if param_name in required_fields else None
            field_defs[param_name] = (param_type, Field(default, description=param_def.get("description", "")))

        if field_defs:
            args_model = create_model(f"{name}_args", **field_defs)
        else:
            args_model = None

        def _make_caller(sn, tn):
            def caller(**kwargs):
                import asyncio
                from .mcp_client import get_mcp_manager
                manager = get_mcp_manager()
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(manager.call_tool(sn, tn, kwargs))
                if result.get("error"):
                    return f"错误: {result['error']}"
                return result.get("content", "")
            return caller

        tool_obj = StructuredTool(
            name=name,
            description=description,
            func=_make_caller(server_name, name),
            args_schema=args_model,
        )
        tools.append(tool_obj)

    return tools


def _json_type_to_python(json_type: str) -> type:
    """JSON Schema 类型映射到 Python 类型。"""
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, str)


def get_dynamic_tools(server_name: str = None) -> list:
    """获取动态 MCP 工具。

    Args:
        server_name: 指定 Server 名称，None 则返回所有

    Returns:
        LangChain Tool 对象列表
    """
    from .mcp_client import get_mcp_manager
    manager = get_mcp_manager()

    all_tools = []
    if server_name:
        tools = manager.get_available_tools(server_name)
        all_tools.extend(mcp_tools_to_langchain_tools(tools, server_name))
    else:
        for name, tools in manager.get_all_tools().items():
            all_tools.extend(mcp_tools_to_langchain_tools(tools, name))

    return all_tools
