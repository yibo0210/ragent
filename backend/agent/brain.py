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
import aiohttp
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
    def _sessions_cache_key(tenant_id: int = None) -> str:
        return f"chat_sessions:{tenant_id or 'anonymous'}"

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

    def save(self, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None, tenant_id: int = None):
        """保存对话"""
        db = SessionLocal()
        try:
            session = (
                db.query(ChatSession)
                .filter(ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                create_kwargs = {"session_id": session_id, "metadata_json": metadata or {}}
                if tenant_id is not None:
                    create_kwargs["tenant_id"] = tenant_id
                session = ChatSession(**create_kwargs)
                db.add(session)
                db.flush()
            else:
                session.metadata_json = metadata or {}
                if tenant_id is not None:
                    session.tenant_id = tenant_id

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
            cache.delete(self._sessions_cache_key(tenant_id))
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

    def list_session_infos(self, tenant_id: int = None) -> list[dict]:
        cached = cache.get_json(self._sessions_cache_key(tenant_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            from sqlalchemy import func
            query = db.query(ChatSession)
            if tenant_id is not None:
                query = query.filter(ChatSession.tenant_id == tenant_id)
            sessions = query.order_by(ChatSession.updated_at.desc()).all()
            if not sessions:
                return []

            # Single-pass: batch-fetch message counts and first messages
            session_ids = [s.id for s in sessions]
            counts = dict(
                db.query(ChatMessage.session_ref_id, func.count(ChatMessage.id))
                .filter(ChatMessage.session_ref_id.in_(session_ids))
                .group_by(ChatMessage.session_ref_id)
                .all()
            )
            first_msgs = dict(
                db.query(ChatMessage.session_ref_id, func.min(ChatMessage.id))
                .filter(ChatMessage.session_ref_id.in_(session_ids), ChatMessage.message_type == "human")
                .group_by(ChatMessage.session_ref_id)
                .all()
            )
            first_msg_ids = [v for v in first_msgs.values()]
            msg_contents = {}
            if first_msg_ids:
                msgs = db.query(ChatMessage).filter(ChatMessage.id.in_(first_msg_ids)).all()
                msg_contents = {m.id: m.content[:20] if m.content else "" for m in msgs}

            result = []
            for s in sessions:
                first_id = first_msgs.get(s.id)
                result.append({
                    "session_id": s.session_id,
                    "updated_at": s.updated_at.isoformat(),
                    "message_count": counts.get(s.id, 0),
                    "first_message": msg_contents.get(first_id, "") if first_id else "",
                })
            cache.set_json(self._sessions_cache_key(tenant_id), result)
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

    def delete_session(self, session_id: str, tenant_id: int = None) -> bool:
        """删除指定会话（按租户隔离），返回是否删除成功"""
        db = SessionLocal()
        try:
            query = db.query(ChatSession).filter(ChatSession.session_id == session_id)
            if tenant_id is not None:
                query = query.filter(ChatSession.tenant_id == tenant_id)
            session = query.first()
            if not session:
                return False

            t_id = session.tenant_id
            db.delete(session)
            db.commit()
            cache.delete(self._messages_cache_key(session_id))
            cache.delete(self._sessions_cache_key(t_id))
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


async def _notify_hitl_webhook(tenant_id: int, interrupt_data: dict):
    """Send webhook notification to tenant admin for HITL approval."""
    webhook_url = os.getenv("HITL_WEBHOOK_URL")
    if not webhook_url:
        return
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(
                webhook_url,
                json={
                    "event": "hitl_interrupt",
                    "tenant_id": tenant_id,
                    "interrupt_type": interrupt_data.get("type"),
                    "message": interrupt_data.get("message"),
                },
                timeout=aiohttp.ClientTimeout(total=5),
            )
    except Exception as e:
        log.warning("hitl_webhook_failed", tenant_id=tenant_id, error=str(e))


# ---------------------------------------------------------------------------
# 非流式对话
# ---------------------------------------------------------------------------
def chat_with_agent(user_text: str, session_id: str = "default_session", user_context: dict = None):
    """使用 Supervisor 多智能体处理用户消息并返回响应。"""
    # --- v6.0 语义缓存 ---
    try:
        from backend.cache import query_cache as cache_lookup
        cached = cache_lookup(user_text)
        if cached:
            return {"response": cached["response"], "cached": True,
                    "source": "semantic_cache"}
    except Exception as e:
        log.debug("semantic_cache_lookup_failed", error=str(e))
    # ---

    messages, _ = _prepare_messages(session_id, user_text)

    # 调用 Supervisor 图
    graph = _get_supervisor_graph()
    result = graph.invoke(
        {"messages": messages, "user_query": user_text, "user_context": user_context or {}},
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
    storage.save(session_id, messages, extra_message_data=extra_message_data,
                 tenant_id=(user_context or {}).get("tenant_id"))

    # v19: Memory extraction after conversation save
    from backend.config import get_settings
    if get_settings().memory_enabled and user_context:
        try:
            import asyncio
            from backend.memory.extractor import get_memory_extractor
            from backend.memory.store import get_memory_store
            extractor = get_memory_extractor()
            store = get_memory_store()
            async def _extract_memories():
                extraction = await extractor.extract(
                    messages=messages,
                    user_id=user_context.get("user_id", 0),
                    tenant_id=user_context.get("tenant_id", 0),
                    session_id=session_id,
                )
                for mem in extraction.memories:
                    store.save(mem)
            asyncio.create_task(_extract_memories())
        except Exception:
            pass

    return {
        "response": response_content,
        "rag_trace": rag_trace,
        "agent_trace": agent_trace,
    }


# ---------------------------------------------------------------------------
# 流式对话（SSE）
# ---------------------------------------------------------------------------
async def chat_with_agent_stream(user_text: str, session_id: str = "default_session", user_context: dict = None):
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
                {"messages": messages, "user_query": user_text, "user_context": user_context or {}},
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
                    asyncio.create_task(
                        _notify_hitl_webhook(
                            (user_context or {}).get("tenant_id", 0),
                            interrupt_info,
                        )
                    )
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
                        # v12: Query Profiler 事件
                        if update.get("query_intent"):
                            await output_queue.put({
                                "type": "query_profiler",
                                "intent": update["query_intent"],
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
                            # v8: 保存 draft_answer
                            if update.get("draft_answer"):
                                full_response = update["draft_answer"]
                        # 单 Worker 情形：synthesize 返回空 {} → update=None
                        if not full_response:
                            pass

                    elif node_name == "planner":
                        # v8: Planner 节点
                        if update is None:
                            continue
                        plan = update.get("query_plan", {})
                        await output_queue.put({
                            "type": "agent_start",
                            "agent": "planner",
                            "timestamp": asyncio.get_event_loop().time(),
                        })
                        await output_queue.put({
                            "type": "plan_generated",
                            "plan": plan,
                            "reasoning": plan.get("reasoning", ""),
                            "steps": plan.get("steps", []),
                        })
                        await output_queue.put({
                            "type": "agent_done",
                            "agent": "planner",
                            "timestamp": asyncio.get_event_loop().time(),
                        })

                    elif node_name == "critique":
                        # v8: Critique 节点
                        if update is None:
                            continue
                        result = update.get("critique_result", {})
                        await output_queue.put({
                            "type": "agent_start",
                            "agent": "critique",
                            "timestamp": asyncio.get_event_loop().time(),
                        })
                        await output_queue.put({
                            "type": "critique_feedback",
                            "is_valid": result.get("is_valid", True),
                            "feedback": result.get("feedback", ""),
                            "missing_information": result.get("missing_information", []),
                        })
                        if not result.get("is_valid", True):
                            await output_queue.put({
                                "type": "self_correction",
                                "message": f"检测到依据不足，正在补充信息...",
                            })
                        await output_queue.put({
                            "type": "agent_done",
                            "agent": "critique",
                            "timestamp": asyncio.get_event_loop().time(),
                        })

                    elif node_name == "replan":
                        # v8: Replan 节点
                        retry = update.get("retry_count", 0) if update else 0
                        await output_queue.put({
                            "type": "rag_step",
                            "step": {
                                "icon": "🔄",
                                "label": f"自纠错 — 根据核查反馈重新规划 (重试 {retry}/2)",
                                "agent": "critique",
                            },
                        })

                    # --- v16: Workflow node events (if workflow graph runs in same stream) ---
                    elif node_name in ("init", "execute_step", "finalize", "handle_error"):
                        if update is not None:
                            await output_queue.put({
                                "type": "workflow_event",
                                "node": node_name,
                                "status": update.get("status", ""),
                                "progress": update.get("progress", 0),
                            })

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
    storage.save(session_id, messages, extra_message_data=extra_message_data,
                 tenant_id=(user_context or {}).get("tenant_id"))

    # v19: Memory extraction after streaming conversation save
    from backend.config import get_settings
    if get_settings().memory_enabled and user_context:
        try:
            from backend.memory.extractor import get_memory_extractor
            from backend.memory.store import get_memory_store
            extractor = get_memory_extractor()
            store = get_memory_store()
            extraction = await extractor.extract(
                messages=messages,
                user_id=user_context.get("user_id", 0),
                tenant_id=user_context.get("tenant_id", 0),
                session_id=session_id,
            )
            for mem in extraction.memories:
                store.save(mem)
        except Exception:
            pass

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
