# v26 Decentralized Agent Network 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for tracking.

**Goal:** 将当前 Supervisor→Workers 的集中式调度架构升级为去中心化 Agent Network，使 Agent 之间可以点对点通信、主动请求帮助、共享证据/记忆/策略。Agent 不再只是被 Supervisor 调度的 Worker，而是拥有自主协商能力的对等节点。

**Architecture:** 新增 `backend/agent_bus/` 包，包含 4 个模块。Mailbox 为每个 Agent 提供异步消息队列（Redis List），Agent 可以发送/接收/轮询消息。MessageRouter 根据消息类型（help_request/evidence_share/memory_share/policy_share/result_notify）和内容语义路由到最合适的接收 Agent。EventBus 基于 Redis Pub/Sub 实现 Agent 间事件广播（task_started/task_completed/evidence_found/conflict_detected），Agent 按需订阅事件频道。Negotiation 支持 Agent 之间的简单协商协议（请求→报价→接受/拒绝），用于任务分解和资源分配。Supervisor 保留为 fallback 协调器（当 Agent 自主协商失败时回退）。复用 v24 World State 的全局状态感知。

**Tech Stack:** Redis 7.0（Mailbox List + EventBus Pub/Sub）· Pydantic v2 · asyncio · 复用 SupervisorState · 复用 v24 World State

---

## File Structure

```
backend/agent_bus/                         # 新增包
├── __init__.py                            # 包入口 + 单例工厂
├── schemas.py                             # AgentMessage, MessageType, Event, NegotiationOffer
├── mailbox.py                             # 每个 Agent 的异步消息队列
├── message_router.py                      # 语义路由到最合适的 Agent
├── event_bus.py                           # Redis Pub/Sub 事件广播
├── negotiation.py                         # Agent 间协商协议

backend/agent/orchestrator.py              # 修改: 支持 Agent 自主请求帮助 + fallback

tests/test_agent_bus.py                    # 新增: Agent Network 测试
```

---

## Phase 1: Schemas + Mailbox

### Task 1: Agent Bus Schemas

**Files:**
- Create: `backend/agent_bus/__init__.py`
- Create: `backend/agent_bus/schemas.py`

```python
# backend/agent_bus/schemas.py
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class MessageType(str, Enum):
    HELP_REQUEST = "help_request"           # "我需要帮助分析这些数据"
    EVIDENCE_SHARE = "evidence_share"       # "我找到了相关证据"
    MEMORY_SHARE = "memory_share"           # "我记得上次类似任务的做法"
    POLICY_SHARE = "policy_share"           # "我有一条执行策略建议"
    RESULT_NOTIFY = "result_notify"         # "我的子任务完成了"


class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AgentMessage(BaseModel):
    """A message sent between agents via the bus."""
    message_id: str = ""
    from_agent: str = ""                   # agent name: "web_searcher", "data_analyst", etc.
    to_agent: str = ""                     # specific agent or "" for broadcast
    message_type: MessageType = MessageType.HELP_REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    subject: str = ""                      # one-line summary
    body: dict = Field(default_factory=dict)  # structured payload
    reply_to: str = ""                     # message_id this replies to
    ttl_seconds: int = 300
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentEvent(BaseModel):
    """A broadcast event via the EventBus."""
    event_id: str = ""
    event_type: str = ""                   # "task_started", "evidence_found", "conflict_detected"
    source_agent: str = ""
    payload: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class NegotiationOffer(BaseModel):
    """An offer in an agent negotiation."""
    offer_id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    task_description: str = ""
    estimated_cost: str = ""               # "3 LLM calls + 2 Tavily searches"
    deadline_seconds: int = 120
    status: str = "pending"                # pending → accepted → completed | rejected | expired
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
```

### Task 2: Mailbox + Message Router

**Files:**
- Create: `backend/agent_bus/mailbox.py`
- Create: `backend/agent_bus/message_router.py`

```python
# backend/agent_bus/mailbox.py
"""Mailbox: per-agent async message queue backed by Redis List."""

import json
from backend.storage.cache import get_redis

_MAILBOX_KEY = "agent_mailbox:{agent_name}"
_MAILBOX_TTL = 600  # 10 min TTL for unread messages

class Mailbox:
    def send(self, message: AgentMessage):
        """Push message to recipient's mailbox (Redis LPUSH)."""
        r = get_redis()
        key = _MAILBOX_KEY.format(agent_name=message.to_agent)
        r.lpush(key, message.model_dump_json())
        r.expire(key, _MAILBOX_TTL)

    def receive(self, agent_name: str, limit: int = 5) -> list[AgentMessage]:
        """Pop messages from agent's mailbox (Redis RPOP in batch)."""
        ...

    def peek(self, agent_name: str, limit: int = 10) -> list[AgentMessage]:
        """Read messages without removing (LRANGE)."""
        ...

    def get_unread_count(self, agent_name: str) -> int:
        ...


# backend/agent_bus/message_router.py
"""MessageRouter: routes messages to the most suitable agent(s)."""

# Agent capability descriptions for semantic routing
_AGENT_CAPABILITIES = {
    "web_searcher": "web search, real-time information, external data",
    "data_analyst": "SQL queries, structured data analysis, MCP data sources",
    "rag_specialist": "internal document retrieval, knowledge base search",
    "local_graph_search": "knowledge graph traversal, entity relationships, multi-hop reasoning",
    "global_graph_search": "community summaries, global overview, macro patterns",
    "direct_answer": "simple Q&A, chitchat, general knowledge",
}

class MessageRouter:
    def route(self, message: AgentMessage) -> list[str]:
        """Determine which agent(s) should receive a broadcast message.
        For HELP_REQUEST: match against agent capabilities via keyword + LLM.
        For EVIDENCE_SHARE: send to agents working on related tasks.
        """
        ...

    def find_best_agent(self, help_request: str) -> str:
        """Find the single best agent for a help request."""
        ...
```

---

## Phase 2: Event Bus + Negotiation

### Task 3: Event Bus + Negotiation

**Files:**
- Create: `backend/agent_bus/event_bus.py`
- Create: `backend/agent_bus/negotiation.py`

```python
# backend/agent_bus/event_bus.py
"""EventBus: Redis Pub/Sub based agent event broadcasting."""

import json
import asyncio
from backend.storage.cache import get_redis

_AGENT_CHANNEL_PREFIX = "agent_events"

class EventBus:
    def publish(self, event: AgentEvent):
        """Publish event to Redis channel. Agents subscribe to relevant channels."""
        r = get_redis()
        channel = f"{_AGENT_CHANNEL_PREFIX}:{event.event_type}"
        r.publish(channel, event.model_dump_json())

    async def subscribe(self, agent_name: str, event_types: list[str]):
        """Subscribe agent to specific event types (long-lived async listener)."""
        ...

    async def listen(self, agent_name: str, callback) -> None:
        """Start listening for subscribed events, invoke callback on each."""
        ...


# backend/agent_bus/negotiation.py
"""Negotiation: simple request-offer-accept protocol between agents."""

class Negotiation:
    def request_help(self, from_agent: str, task: str, context: dict) -> NegotiationOffer:
        """Send a help request and wait for offers."""
        ...

    def make_offer(self, offer: NegotiationOffer) -> bool:
        """Respond to a help request with an offer."""
        ...

    def accept_offer(self, offer_id: str) -> bool:
        """Accept an offer and assign the task."""
        ...

    def reject_offer(self, offer_id: str, reason: str = "") -> bool:
        ...

    def wait_for_offers(self, request_id: str, timeout: int = 30) -> list[NegotiationOffer]:
        """Wait for agents to respond with offers (async with timeout)."""
        ...
```

---

## Phase 3: Agent Self-Dispatch

### Task 4: Agent Self-Dispatch Logic

**Files:**
- Modify: `backend/agent/orchestrator.py`

在每个 Agent Worker 节点中增加自主请求帮助的能力:

```python
# v26: Agent self-dispatch — agents can request help from peers
from backend.agent_bus.mailbox import get_mailbox
from backend.agent_bus.message_router import get_message_router
from backend.agent_bus.event_bus import get_event_bus

# In any agent node, when stuck or needing help:
async def agent_node_with_help(state: SupervisorState):
    # ... do work ...

    if need_help:
        msg = AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            from_agent=self.name,
            message_type=MessageType.HELP_REQUEST,
            priority=MessagePriority.HIGH,
            subject="需要补充外部数据",
            body={"missing": "latest market data", "query": original_query},
        )

        # Try peer-to-peer first
        router = get_message_router()
        best_agent = router.find_best_agent(msg.subject)
        msg.to_agent = best_agent
        get_mailbox().send(msg)

        # Emit event for observability
        get_event_bus().publish(AgentEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type="help_requested",
            source_agent=self.name,
            payload={"to": best_agent, "subject": msg.subject},
        ))

    # Check own mailbox for incoming messages before finishing
    mailbox = get_mailbox()
    incoming = mailbox.receive(self.name, limit=3)
    for msg in incoming:
        if msg.message_type == MessageType.EVIDENCE_SHARE:
            # Incorporate shared evidence into result
            ...
```

Supervisor 保留为 fallback:

```python
# v26: Supervisor becomes fallback coordinator
# Only intervene when:
# 1. Agent negotiation fails (no offers after timeout)
# 2. Circular help requests detected
# 3. New task arrives (initial routing)
```

---

## Phase 4: Shared Memory via Bus

### Task 5: Cross-Agent Memory Sharing

**Files:**
- Modify: `backend/agent_bus/message_router.py`

```python
# v26: Agents share memory/evidence/policy via the bus
# When web_searcher finds a great source:
await event_bus.publish(AgentEvent(
    event_type="evidence_found",
    source_agent="web_searcher",
    payload={"url": source_url, "summary": summary, "confidence": 0.9},
))

# When data_analyst discovers a useful SQL pattern:
await mailbox.send(AgentMessage(
    message_type=MessageType.POLICY_SHARE,
    from_agent="data_analyst",
    body={"policy": "Always run SELECT COUNT(*) before complex joins", "domain": "sql"},
))

# rag_specialist subscribes to evidence_found events
await event_bus.subscribe("rag_specialist", ["evidence_found", "conflict_detected"])
```

---

## Self-Review

| 610 文档 v26 需求 | 覆盖 |
|---|---|
| Mailbox (Agent 消息队列) | Task 2 (mailbox.py) |
| MessageRouter (语义路由) | Task 2 (message_router.py) |
| EventBus (发布/订阅广播) | Task 3 (event_bus.py) |
| Negotiation (协商协议) | Task 3 (negotiation.py) |
| Agent 主动请求帮助 | Task 4 (orchestrator.py agent_node_with_help) |
| 共享证据/记忆/策略 | Task 5 (Cross-Agent Memory Sharing) |
| Supervisor fallback 协调 | Task 4 (Supervisor fallback) |
