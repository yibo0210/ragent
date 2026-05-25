"""AI 对话大脑模块（多智能体版本）

核心功能：
- 多轮对话记忆（MySQL + Redis）
- 流式输出（SSE 实时返回内容/RAG 步骤/路由事件）
- Supervisor-Workers 多智能体编排（LangGraph）
- 长对话自动摘要
"""
from dotenv import load_dotenv
import os
import json
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .tools import get_last_rag_context, reset_tool_call_guards, set_rag_step_queue
from datetime import datetime, timezone
from backend.storage.cache import cache
from backend.storage.database import SessionLocal
from backend.storage.models import ChatSession, ChatMessage

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


class ConversationStorage:
    """对话存储（MySQL + Redis）。
    核心功能：对话的持久化、加载、缓存、删除，整合 MySQL（持久化）和 Redis（缓存），兼顾数据可靠性与查询性能。"""

    @staticmethod
    def _messages_cache_key(session_id: str) -> str:
        return f"chat_messages:{session_id}"

    @staticmethod
    def _sessions_cache_key() -> str:
        return f"chat_sessions:anonymous"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def save(self, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        """保存对话"""
        db = SessionLocal()
        try:
            session = (
                db.query(ChatSession)
                .filter(ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                session = ChatSession(session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                db.flush()
            else:
                session.metadata_json = metadata or {}

            db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).delete(synchronize_session=False)

            serialized = []
            now = datetime.now(timezone.utc)
            for idx, msg in enumerate(messages):
                rag_trace = None
                agent_trace = None
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = extra.get("rag_trace")
                    agent_trace = extra.get("agent_trace")

                db.add(
                    ChatMessage(
                        session_ref_id=session.id,
                        message_type=msg.type,
                        content=str(msg.content),
                        timestamp=now,
                        rag_trace=rag_trace,
                        agent_trace=agent_trace,
                    )
                )
                serialized.append(
                    {
                        "type": msg.type,
                        "content": str(msg.content),
                        "timestamp": now.isoformat(),
                        "rag_trace": rag_trace,
                        "agent_trace": agent_trace,
                    }
                )

            session.updated_at = now
            db.commit()

            cache.set_json(self._messages_cache_key(session_id), serialized)
            cache.delete(self._sessions_cache_key())
        finally:
            db.close()

    def load(self, session_id: str) -> list:
        """加载对话"""
        cached = cache.get_json(self._messages_cache_key(session_id))
        if cached is not None:
            return self._to_langchain_messages(cached)

        records = self.get_session_messages(session_id)
        cache.set_json(self._messages_cache_key(session_id), records)
        return self._to_langchain_messages(records)

    def list_sessions(self) -> list:
        """列出所有会话"""
        return [item["session_id"] for item in self.list_session_infos()]

    def list_session_infos(self) -> list[dict]:
        cached = cache.get_json(self._sessions_cache_key())
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            sessions = (
                db.query(ChatSession)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = []
            for s in sessions:
                count = db.query(ChatMessage).filter(ChatMessage.session_ref_id == s.id).count()
                first_msg = (
                    db.query(ChatMessage)
                    .filter(ChatMessage.session_ref_id == s.id, ChatMessage.message_type == "human")
                    .order_by(ChatMessage.id.asc())
                    .first()
                )
                result.append(
                    {
                        "session_id": s.session_id,
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count,
                        "first_message": first_msg.content[:20] if first_msg else "",
                    }
                )
            cache.set_json(self._sessions_cache_key(), result)
            return result
        finally:
            db.close()

    def get_session_messages(self, session_id: str) -> list[dict]:
        cached = cache.get_json(self._messages_cache_key(session_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            session = (
                db.query(ChatSession)
                .filter(ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []

            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            result = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace,
                    "agent_trace": getattr(row, "agent_trace", None),
                }
                for row in rows
            ]
            cache.set_json(self._messages_cache_key(session_id), result)
            return result
        finally:
            db.close()

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话，返回是否删除成功"""
        db = SessionLocal()
        try:
            session = (
                db.query(ChatSession)
                .filter(ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False

            db.delete(session)
            db.commit()
            cache.delete(self._messages_cache_key(session_id))
            cache.delete(self._sessions_cache_key())
            return True
        finally:
            db.close()


# ---------------------------------------------------------------------------
# 对话摘要
# ---------------------------------------------------------------------------
def summarize_old_messages(model, messages: list) -> str:
    """将旧消息总结为摘要"""
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    summary_prompt = f"""请总结以下对话的关键信息：

{old_conversation}
总结（包含用户信息、重要事实、待办事项）："""

    summary = model.invoke(summary_prompt).content
    return summary


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------
storage = ConversationStorage()

# 导入 Supervisor 图（模块级单例）
from .orchestrator import _get_supervisor_graph

# 用于摘要的模型
_summary_model = None


def _get_summary_model():
    global _summary_model
    if _summary_model is None:
        _summary_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.3,
            timeout=60,
        )
    return _summary_model


def _prepare_messages(session_id: str, user_text: str) -> tuple[list, bool]:
    """加载对话历史，处理摘要，返回 (messages, need_summary)。"""
    messages = storage.load(session_id)
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    need_summary = len(messages) > 50
    if need_summary:
        summary_model = _get_summary_model()
        summary = summarize_old_messages(summary_model, messages[:40])
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    messages.append(HumanMessage(content=user_text))
    return messages, need_summary


# ---------------------------------------------------------------------------
# 非流式对话
# ---------------------------------------------------------------------------
def chat_with_agent(user_text: str, session_id: str = "default_session"):
    """使用 Supervisor 多智能体处理用户消息并返回响应。"""
    messages, _ = _prepare_messages(session_id, user_text)

    # 调用 Supervisor 图
    graph = _get_supervisor_graph()
    result = graph.invoke(
        {"messages": messages, "user_query": user_text},
        config={"configurable": {"thread_id": session_id}, "recursion_limit": 15},
    )

    # 提取回答
    response_content = ""
    result_messages = result.get("messages", [])
    if result_messages:
        last_msg = result_messages[-1]
        response_content = getattr(last_msg, "content", str(last_msg))

    # 提取 traces
    rag_trace = result.get("rag_trace")
    agent_trace = result.get("agent_trace")

    # 保存对话
    messages.append(AIMessage(content=response_content))
    extra_message_data = [None] * (len(messages) - 1) + [{
        "rag_trace": rag_trace,
        "agent_trace": agent_trace,
    }]
    storage.save(session_id, messages, extra_message_data=extra_message_data)

    return {
        "response": response_content,
        "rag_trace": rag_trace,
        "agent_trace": agent_trace,
    }


# ---------------------------------------------------------------------------
# 流式对话（SSE）
# ---------------------------------------------------------------------------
async def chat_with_agent_stream(user_text: str, session_id: str = "default_session"):
    """使用 Supervisor 多智能体处理用户消息并流式返回响应。

    SSE 事件协议：
    - {"type": "routing", "agent": "...", "reason": "..."}  -- 路由决策
    - {"type": "rag_step", "step": {...}}                    -- 检索步骤
    - {"type": "content", "content": "..."}                  -- 回答内容
    - {"type": "trace", "rag_trace": {...}}                  -- RAG 追踪
    - {"type": "agent_trace", "agent_trace": {...}}          -- Agent 追踪
    - {"type": "error", "content": "..."}                    -- 错误
    - data: [DONE]                                           -- 结束
    """
    messages, _ = _prepare_messages(session_id, user_text)

    # 统一输出队列
    output_queue = asyncio.Queue()

    # 设置 RAG 步骤队列（复用现有机制）
    class _RagStepProxy:
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})
            agent = step.get("agent", "")
            if agent == "local_graph_search":
                output_queue.put_nowait({
                    "type": "graph_expand",
                    "agent": agent,
                    "message": f"{step.get('label', '')}",
                })
            elif agent == "global_graph_search":
                output_queue.put_nowait({
                    "type": "community_match",
                    "agent": agent,
                    "message": f"{step.get('label', '')}",
                })

    set_rag_step_queue(_RagStepProxy())

    # 设置 token 流式传输队列
    from .tools import set_token_queue as _set_token_queue

    class _TokenProxy:
        def put_nowait(self, event):
            output_queue.put_nowait(event)

    _set_token_queue(_TokenProxy())

    full_response = ""
    final_rag_trace = None
    final_agent_trace = None

    interrupt_info = None  # 用于 HITL 中断标记

    async def _graph_worker():
        """后台任务：运行 Supervisor 图并将事件推入输出队列。"""
        nonlocal full_response, final_rag_trace, final_agent_trace, interrupt_info
        try:
            graph = _get_supervisor_graph()
            async for event in graph.astream(
                {"messages": messages, "user_query": user_text},
                stream_mode="updates",
                config={"configurable": {"thread_id": session_id}, "recursion_limit": 15},
            ):
                # 检查中断事件
                if "__interrupt__" in event:
                    interrupt_data = event["__interrupt__"]
                    actual = interrupt_data[0] if isinstance(interrupt_data, tuple) else interrupt_data
                    interrupt_info = actual if isinstance(actual, dict) else {"data": str(actual)}
                    # 加锁防并发消息
                    cache.acquire_lock(session_id)
                    await output_queue.put({
                        "type": "hitl_interrupt",
                        "data": interrupt_info,
                    })
                    break

                # event 格式: {"node_name": state_update_dict}
                for node_name, update in event.items():
                    if node_name == "supervisor":
                        if update is None:
                            continue
                        # 路由决策事件
                        route = update.get("next_worker", "")
                        reason = update.get("route_reason", "")
                        next_workers = update.get("next_workers", [route] if route else [])
                        await output_queue.put({
                            "type": "agent_start",
                            "agent": "supervisor",
                            "timestamp": asyncio.get_event_loop().time(),
                        })
                        if route:
                            await output_queue.put({
                                "type": "routing",
                                "agent": route,
                                "reason": reason,
                            })
                        await output_queue.put({
                            "type": "agent_done",
                            "agent": "supervisor",
                            "timestamp": asyncio.get_event_loop().time(),
                        })
                        # worker agent_start 提前到路由时发送，trace 面板可实时显示活跃 agent
                        for worker in next_workers:
                            if worker != "supervisor":
                                await output_queue.put({
                                    "type": "agent_start",
                                    "agent": worker,
                                    "timestamp": asyncio.get_event_loop().time(),
                                })

                    elif node_name in ("rag_specialist", "web_searcher", "data_analyst", "direct_answer", "local_graph_search", "global_graph_search"):
                        # Worker 完成，提取回答内容
                        result_messages = update.get("messages", [])
                        if result_messages:
                            last_msg = result_messages[-1]
                            content = getattr(last_msg, "content", "")
                            if content:
                                full_response = content
                                await output_queue.put({
                                    "type": "worker_content",
                                    "agent": node_name,
                                    "content": content,
                                })

                        # 提取 traces
                        if update.get("rag_trace"):
                            final_rag_trace = update["rag_trace"]
                        if update.get("agent_trace"):
                            final_agent_trace = update["agent_trace"]

                        await output_queue.put({
                            "type": "agent_done",
                            "agent": node_name,
                            "timestamp": asyncio.get_event_loop().time(),
                        })

                    elif node_name == "synthesize":
                        # Synthesize 节点：有多 worker 聚合时发送合并结果（token 已流式推送）
                        if update is not None:
                            result_messages = update.get("messages", [])
                            if result_messages:
                                last_msg = result_messages[-1]
                                content = getattr(last_msg, "content", "")
                                if content:
                                    full_response = content
                        # 单 Worker 情形：synthesize 返回空 {} → update=None
                        if not full_response:
                            pass

        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            # 哨兵：通知主循环图执行完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_graph_worker())

    try:
        # 主循环：持续从队列取事件并 yield SSE
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        # 客户端断开连接
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        # 清理
        set_rag_step_queue(None)
        _set_token_queue(None)
        if not agent_task.done():
            agent_task.cancel()

    # 发送 trace 事件
    if final_rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': final_rag_trace})}\n\n"
    if final_agent_trace:
        yield f"data: {json.dumps({'type': 'agent_trace', 'agent_trace': final_agent_trace})}\n\n"

    # HITL 中断时不发送 [DONE] 也不保存对话
    if interrupt_info:
        set_rag_step_queue(None)
        _set_token_queue(None)
        return

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话
    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{
        "rag_trace": final_rag_trace,
        "agent_trace": final_agent_trace,
    }]
    storage.save(session_id, messages, extra_message_data=extra_message_data)


# ---------------------------------------------------------------------------
# HITL 恢复函数
# ---------------------------------------------------------------------------
async def resume_hitl_graph(session_id: str, action: str, modified_input: str = ""):
    """恢复因 HITL 中断而挂起的图执行。"""
    from langgraph.types import Command
    from .tools import set_rag_step_queue, set_token_queue as _set_token_queue

    output_queue = asyncio.Queue()
    cache.release_lock(session_id)

    class _RagStepProxy:
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})
            agent = step.get("agent", "")
            if agent == "local_graph_search":
                output_queue.put_nowait({
                    "type": "graph_expand",
                    "agent": agent,
                    "message": f"{step.get('label', '')}",
                })
            elif agent == "global_graph_search":
                output_queue.put_nowait({
                    "type": "community_match",
                    "agent": agent,
                    "message": f"{step.get('label', '')}",
                })
    set_rag_step_queue(_RagStepProxy())

    class _TokenProxy:
        def put_nowait(self, event):
            output_queue.put_nowait(event)
    _set_token_queue(_TokenProxy())

    resume_value = {"action": action}
    if action == "modify" and modified_input:
        resume_value["human_interfered_input"] = modified_input

    full_response = ""
    final_rag_trace = None
    final_agent_trace = None

    async def _resume_worker():
        nonlocal full_response, final_rag_trace, final_agent_trace
        try:
            graph = _get_supervisor_graph()
            command = Command(resume=resume_value)
            async for event in graph.astream(
                command,
                stream_mode="updates",
                config={"configurable": {"thread_id": session_id}, "recursion_limit": 15},
            ):
                for node_name, update in event.items():
                    if node_name in ("rag_specialist", "web_searcher", "data_analyst", "direct_answer", "local_graph_search", "global_graph_search"):
                        result_messages = update.get("messages", [])
                        if result_messages:
                            last_msg = result_messages[-1]
                            content = getattr(last_msg, "content", "")
                            if content:
                                full_response = content
                        if update.get("rag_trace"):
                            final_rag_trace = update["rag_trace"]
                        if update.get("agent_trace"):
                            final_agent_trace = update["agent_trace"]
                    elif node_name == "synthesize":
                        result_messages = update.get("messages", [])
                        if result_messages:
                            last_msg = result_messages[-1]
                            content = getattr(last_msg, "content", "")
                            if content:
                                full_response = content
                        elif full_response:
                            pass
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            await output_queue.put(None)

    agent_task = asyncio.create_task(_resume_worker())

    try:
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except GeneratorExit:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        set_rag_step_queue(None)
        _set_token_queue(None)
        if not agent_task.done():
            agent_task.cancel()

    if final_rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': final_rag_trace}, ensure_ascii=False)}\n\n"
    if final_agent_trace:
        yield f"data: {json.dumps({'type': 'agent_trace', 'agent_trace': final_agent_trace}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
