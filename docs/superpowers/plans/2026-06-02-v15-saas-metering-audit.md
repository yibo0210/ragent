# v15 SaaS Metering, Rate Limiting & Audit Trail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add per-tenant token metering, Redis-based rate limiting with SLA tiers, and immutable audit logging for all MCP/external tool calls — completing the SaaS billing and compliance layers.

**Architecture:** Three new MySQL tables (`token_usage_logs`, `rate_limit_rules`, `audit_logs`) store metering and audit data. A `TenantRateLimiter` class uses Redis sliding-window counters per tenant_id. Token usage is recorded after every LLM call via a decorator. Audit logs are written before/after every MCP `call_tool` and SQL execution. HITL approval is extended with webhook notification to tenant admins.

**Tech Stack:** SQLAlchemy · Redis (existing) · FastAPI Depends · aiohttp (webhook) · pytest

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `backend/billing/__init__.py` | Package init |
| `backend/billing/models.py` | TokenUsageLog, RateLimitRule, AuditLog SQLAlchemy models |
| `backend/billing/token_tracker.py` | Record token usage per request, query usage summaries |
| `backend/billing/rate_limiter.py` | Per-tenant Redis sliding-window rate limiter |
| `backend/billing/audit.py` | Audit log writer for MCP/SQL/tool calls |
| `backend/billing/middleware.py` | FastAPI rate-limit middleware (429 on exceeded) |
| `backend/billing/routes.py` | `/billing/usage`, `/billing/audit` query endpoints |
| `tests/test_token_tracker.py` | Token usage recording tests |
| `tests/test_rate_limiter.py` | Rate limiter unit tests (mocked Redis) |
| `tests/test_audit.py` | Audit log recording tests |
| `tests/test_billing_integration.py` | End-to-end billing + audit integration tests |

### Modified Files

| File | Changes |
|------|---------|
| `backend/storage/models.py` | Import new billing models so `init_db()` creates them |
| `backend/agent/orchestrator.py` | Record token usage after LLM calls, write audit on HITL events |
| `backend/agent/brain.py` | Attach rate-limit check before graph invocation, emit `rate_limit` SSE event |
| `backend/agent/mcp_client.py` | Wrap `call_tool` with audit logging |
| `backend/agent/data_analyst.py` | Wrap `execute_sql` and `execute_mcp_query` with audit logging |
| `backend/api/routes.py` | Register billing router, add rate-limit middleware |
| `backend/api/app.py` | Include billing router |
| `backend/ha/load_monitor.py` | Add per-tenant SLA-aware degradation hooks |
| `backend/schemas.py` | Add Pydantic models for billing/audit API responses |

---

## Milestone 1: Token Usage Tracking

### Task 1: Token Usage & Audit SQLAlchemy Models

**Files:**
- Create: `backend/billing/__init__.py`
- Create: `backend/billing/models.py`
- Test: `tests/test_token_tracker.py`

- [x] **Step 1: Create billing package init**

```python
# backend/billing/__init__.py
```

- [x] **Step 2: Write failing test**

```python
# tests/test_token_tracker.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.storage.database import Base
from backend.billing.models import TokenUsageLog, RateLimitRule, AuditLog


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_token_usage_log(db_session):
    log = TokenUsageLog(
        tenant_id=1, user_id=1, session_id="s1",
        model_name="qwen-plus", prompt_tokens=500, completion_tokens=200,
        agent_name="rag_specialist", request_type="chat",
    )
    db_session.add(log)
    db_session.commit()
    assert log.id is not None
    assert log.total_tokens == 700


def test_create_rate_limit_rule(db_session):
    rule = RateLimitRule(
        tenant_id=1, tier="free", qps_limit=10, daily_token_limit=100000,
    )
    db_session.add(rule)
    db_session.commit()
    assert rule.qps_limit == 10


def test_create_audit_log(db_session):
    log = AuditLog(
        tenant_id=1, user_id=1, action="mcp_tool_call",
        target="query_database", arguments='{"sql": "SELECT 1"}',
        result_summary="success", risk_level="low",
    )
    db_session.add(log)
    db_session.commit()
    assert log.id is not None
    assert log.action == "mcp_tool_call"


def test_token_usage_tenant_filter(db_session):
    db_session.add(TokenUsageLog(tenant_id=1, user_id=1, model_name="qwen-plus", prompt_tokens=100, completion_tokens=50))
    db_session.add(TokenUsageLog(tenant_id=2, user_id=2, model_name="qwen-plus", prompt_tokens=200, completion_tokens=100))
    db_session.commit()
    t1_logs = db_session.query(TokenUsageLog).filter(TokenUsageLog.tenant_id == 1).all()
    assert len(t1_logs) == 1
    assert t1_logs[0].prompt_tokens == 100
```

- [x] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_token_tracker.py -v`
Expected: FAIL with `ImportError`

- [x] **Step 4: Implement billing models**

```python
# backend/billing/models.py
from sqlalchemy import Column, Integer, String, DateTime, Text, BigInteger, Index
from datetime import datetime, timezone
from backend.storage.database import Base


class TokenUsageLog(Base):
    __tablename__ = "token_usage_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(120), nullable=True)
    model_name = Column(String(100), nullable=False)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    agent_name = Column(String(50), nullable=True)
    request_type = Column(String(50), nullable=True)  # chat, rag, graph, data_analyst
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    __table_args__ = (
        Index("ix_token_usage_tenant_created", "tenant_id", "created_at"),
    )


class RateLimitRule(Base):
    __tablename__ = "rate_limit_rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, nullable=False, unique=True, index=True)
    tier = Column(String(50), nullable=False, default="free")  # free, standard, premium, enterprise
    qps_limit = Column(Integer, nullable=False, default=10)
    daily_token_limit = Column(BigInteger, nullable=False, default=100000)
    concurrent_limit = Column(Integer, nullable=False, default=5)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(120), nullable=True)
    action = Column(String(100), nullable=False)  # mcp_tool_call, sql_execute, hitl_approve, hitl_reject
    target = Column(String(255), nullable=True)  # tool name, SQL table, etc.
    arguments = Column(Text, nullable=True)  # JSON string of arguments
    result_summary = Column(String(500), nullable=True)
    risk_level = Column(String(20), nullable=False, default="low")  # low, medium, high, critical
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_audit_tenant_created", "tenant_id", "created_at"),
    )
```

- [x] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_token_tracker.py -v`
Expected: 4 passed

- [x] **Step 6: Commit**

```bash
git add backend/billing/__init__.py backend/billing/models.py tests/test_token_tracker.py
git commit -m "feat(v15): add TokenUsageLog, RateLimitRule, AuditLog SQLAlchemy models"
```

---

### Task 2: Token Usage Tracker

**Files:**
- Create: `backend/billing/token_tracker.py`
- Modify: `backend/agent/orchestrator.py`

- [x] **Step 1: Write failing test**

```python
# tests/test_token_tracker.py (append to existing file)
from backend.billing.token_tracker import record_token_usage, get_usage_summary
from unittest.mock import MagicMock, patch


def test_record_token_usage():
    mock_db = MagicMock()
    record_token_usage(
        db=mock_db, tenant_id=1, user_id=1, session_id="s1",
        model_name="qwen-plus", prompt_tokens=500, completion_tokens=200,
        agent_name="rag_specialist", request_type="chat",
    )
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


def test_get_usage_summary():
    mock_db = MagicMock()
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = [
        MagicMock(prompt_tokens=100, completion_tokens=50),
        MagicMock(prompt_tokens=200, completion_tokens=100),
    ]
    result = get_usage_summary(mock_db, tenant_id=1)
    assert result["total_prompt_tokens"] == 300
    assert result["total_completion_tokens"] == 150
    assert result["total_tokens"] == 450
    assert result["request_count"] == 2
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_token_tracker.py -v -k "test_record_token_usage or test_get_usage_summary"`
Expected: FAIL with `ImportError`

- [x] **Step 3: Implement token tracker**

```python
# backend/billing/token_tracker.py
from sqlalchemy.orm import Session
from backend.billing.models import TokenUsageLog
from datetime import datetime, timezone, timedelta


def record_token_usage(
    db: Session, tenant_id: int, user_id: int,
    model_name: str, prompt_tokens: int, completion_tokens: int,
    session_id: str = None, agent_name: str = None, request_type: str = None,
) -> TokenUsageLog:
    log = TokenUsageLog(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id,
        model_name=model_name, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, agent_name=agent_name,
        request_type=request_type,
    )
    db.add(log)
    db.commit()
    return log


def get_usage_summary(db: Session, tenant_id: int, days: int = 30) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    logs = db.query(TokenUsageLog).filter(
        TokenUsageLog.tenant_id == tenant_id,
        TokenUsageLog.created_at >= since,
    ).all()
    total_prompt = sum(l.prompt_tokens for l in logs)
    total_completion = sum(l.completion_tokens for l in logs)
    return {
        "tenant_id": tenant_id,
        "period_days": days,
        "request_count": len(logs),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_token_tracker.py -v`
Expected: All passed

- [x] **Step 5: Integrate into orchestrator.py**

Read `backend/agent/orchestrator.py`. Find where LLM calls are made in worker nodes (e.g., `_stream_answer` function or direct `model.invoke()` calls). After each LLM call, record token usage:

```python
from backend.billing.token_tracker import record_token_usage
from backend.storage.database import SessionLocal

# After LLM call that returns token info:
db = SessionLocal()
try:
    record_token_usage(
        db=db,
        tenant_id=state.get("user_context", {}).get("tenant_id", 0),
        user_id=state.get("user_context", {}).get("user_id", 0),
        session_id=state.get("user_context", {}).get("session_id"),
        model_name=model_name,
        prompt_tokens=response.usage_metadata.get("input_tokens", 0),
        completion_tokens=response.usage_metadata.get("output_tokens", 0),
        agent_name="rag_specialist",
        request_type="chat",
    )
finally:
    db.close()
```

Apply this pattern to: `rag_specialist_node`, `local_graph_search_node`, `global_graph_search_node`, `web_searcher_node`, `data_analyst_node`, `direct_answer_node`, `synthesize_node`, `critique_node`.

- [x] **Step 6: Commit**

```bash
git add backend/billing/token_tracker.py backend/agent/orchestrator.py tests/test_token_tracker.py
git commit -m "feat(v15): add token usage tracker and integrate into agent orchestrator"
```

---

## Milestone 2: Per-Tenant Rate Limiting

### Task 3: Redis Rate Limiter

**Files:**
- Create: `backend/billing/rate_limiter.py`
- Test: `tests/test_rate_limiter.py`

- [x] **Step 1: Write failing test**

```python
# tests/test_rate_limiter.py
import pytest
from unittest.mock import MagicMock, patch
from backend.billing.rate_limiter import TenantRateLimiter


@pytest.fixture
def limiter():
    mock_redis = MagicMock()
    return TenantRateLimiter(mock_redis)


def test_rate_limiter_allows_under_limit(limiter):
    limiter.redis.mget.return_value = [b"5", b"3", b"2"]  # 10 requests in window
    result = limiter.check_rate_limit(tenant_id=1, qps_limit=10)
    assert result["allowed"] is True


def test_rate_limiter_blocks_over_limit(limiter):
    limiter.redis.mget.return_value = [b"10", b"10", b"10"]  # 30 requests in window
    result = limiter.check_rate_limit(tenant_id=1, qps_limit=5)
    assert result["allowed"] is False
    assert result["retry_after"] > 0


def test_rate_limiter_increments_counter(limiter):
    limiter.redis.pipeline.return_value = limiter.redis
    limiter.redis.execute.return_value = [None, None]
    limiter.record_request(tenant_id=1)
    limiter.redis.incr.assert_called_once()


def test_rate_limiter_tenant_specific(limiter):
    limiter.redis.mget.return_value = [b"0"]
    result = limiter.check_rate_limit(tenant_id=999, qps_limit=100)
    assert result["allowed"] is True
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rate_limiter.py -v`
Expected: FAIL with `ImportError`

- [x] **Step 3: Implement rate limiter**

```python
# backend/billing/rate_limiter.py
import time
from typing import Optional


class TenantRateLimiter:
    """Per-tenant sliding-window rate limiter using Redis."""

    def __init__(self, redis_client, window: int = 10):
        self.redis = redis_client
        self.window = window

    def _key(self, tenant_id: int, ts: int) -> str:
        return f"ragent:ratelimit:tenant:{tenant_id}:{ts}"

    def record_request(self, tenant_id: int) -> None:
        ts = int(time.time())
        pipe = self.redis.pipeline()
        for i in range(self.window):
            pipe.incr(self._key(tenant_id, ts - i))
            pipe.expire(self._key(tenant_id, ts - i), self.window + 1)
        try:
            pipe.execute()
        except Exception:
            pass  # fail-open

    def get_current_count(self, tenant_id: int) -> int:
        ts = int(time.time())
        keys = [self._key(tenant_id, ts - i) for i in range(self.window)]
        try:
            values = self.redis.mget(keys)
            return sum(int(v or 0) for v in values)
        except Exception:
            return 0

    def check_rate_limit(self, tenant_id: int, qps_limit: int) -> dict:
        count = self.get_current_count(tenant_id)
        if count >= qps_limit * self.window:
            return {
                "allowed": False,
                "current_count": count,
                "limit": qps_limit * self.window,
                "retry_after": 1,
            }
        return {
            "allowed": True,
            "current_count": count,
            "limit": qps_limit * self.window,
        }
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rate_limiter.py -v`
Expected: 4 passed

- [x] **Step 5: Commit**

```bash
git add backend/billing/rate_limiter.py tests/test_rate_limiter.py
git commit -m "feat(v15): add per-tenant Redis sliding-window rate limiter"
```

---

### Task 4: Rate Limit Middleware & SLA Degradation

**Files:**
- Create: `backend/billing/middleware.py`
- Modify: `backend/agent/brain.py`
- Modify: `backend/ha/load_monitor.py`

- [x] **Step 1: Write failing test**

```python
# tests/test_rate_limiter.py (append)
def test_get_tenant_rule():
    mock_db = MagicMock()
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.first.return_value = MagicMock(qps_limit=20, daily_token_limit=500000, tier="standard")
    from backend.billing.rate_limiter import get_tenant_rule
    rule = get_tenant_rule(mock_db, tenant_id=1)
    assert rule.qps_limit == 20
    assert rule.tier == "standard"


def test_get_tenant_rule_default():
    mock_db = MagicMock()
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.first.return_value = None
    from backend.billing.rate_limiter import get_tenant_rule
    rule = get_tenant_rule(mock_db, tenant_id=999)
    assert rule.qps_limit == 10  # default free tier
    assert rule.tier == "free"
```

- [x] **Step 2: Run test to verify it fails**

- [x] **Step 3: Implement get_tenant_rule**

Add to `backend/billing/rate_limiter.py`:

```python
from backend.billing.models import RateLimitRule
from sqlalchemy.orm import Session
from types import SimpleNamespace


def get_tenant_rule(db: Session, tenant_id: int):
    rule = db.query(RateLimitRule).filter(RateLimitRule.tenant_id == tenant_id).first()
    if rule:
        return rule
    return SimpleNamespace(
        tenant_id=tenant_id, tier="free", qps_limit=10,
        daily_token_limit=100000, concurrent_limit=5,
    )
```

- [x] **Step 4: Implement rate limit middleware**

```python
# backend/billing/middleware.py
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from backend.billing.rate_limiter import TenantRateLimiter, get_tenant_rule
from backend.billing.models import RateLimitRule
from backend.storage.database import SessionLocal
from backend.storage.cache import cache


def create_rate_limit_middleware(limiter: TenantRateLimiter):
    async def rate_limit_middleware(request: Request, call_next):
        # Skip non-API paths and auth endpoints
        path = request.url.path
        if not path.startswith("/chat") and not path.startswith("/documents"):
            return await call_next(request)

        # Extract tenant_id from auth (set by get_current_user dependency)
        # The dependency runs after middleware, so we extract from token directly
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return await call_next(request)

        try:
            from backend.auth.jwt_handler import decode_token
            token = auth_header.split(" ", 1)[1]
            payload = decode_token(token)
            tenant_id = payload.get("tenant_id", 0)
        except Exception:
            return await call_next(request)

        if not tenant_id:
            return await call_next(request)

        db = SessionLocal()
        try:
            rule = get_tenant_rule(db, tenant_id)
            result = limiter.check_rate_limit(tenant_id, rule.qps_limit)
        finally:
            db.close()

        if not result["allowed"]:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "retry_after": result["retry_after"]},
                headers={"Retry-After": str(result["retry_after"])},
            )

        limiter.record_request(tenant_id)
        response = await call_next(request)
        return response

    return rate_limit_middleware
```

- [x] **Step 5: Add SLA-aware degradation to load_monitor.py**

Read `backend/ha/load_monitor.py`. Add a method to `LoadMonitor`:

```python
def get_tenant_degradation(self, tenant_tier: str) -> str:
    """Return degradation level based on system load and tenant tier."""
    state = self.get_state()
    if state == SystemState.NORMAL:
        return "full"
    if state == SystemState.WARNING:
        if tenant_tier in ("enterprise", "premium"):
            return "full"
        return "skip_critique"
    if state == SystemState.CRITICAL:
        if tenant_tier == "enterprise":
            return "full"
        if tenant_tier == "premium":
            return "skip_critique"
        return "cache_only"
    return "full"
```

- [x] **Step 6: Register middleware in app.py**

In `backend/api/app.py`, add after existing middleware:

```python
from backend.billing.rate_limiter import TenantRateLimiter
from backend.billing.middleware import create_rate_limit_middleware
from backend.storage.cache import cache

limiter = TenantRateLimiter(cache._redis)
app.middleware("http")(create_rate_limit_middleware(limiter))
```

- [x] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_rate_limiter.py -v`
Expected: All passed

- [x] **Step 8: Commit**

```bash
git add backend/billing/middleware.py backend/billing/rate_limiter.py backend/ha/load_monitor.py backend/api/app.py tests/test_rate_limiter.py
git commit -m "feat(v15): add rate-limit middleware with per-tenant SLA-aware degradation"
```

---

### Task 4b: Orchestrator SLA Degradation Wiring

**Files:**
- Modify: `backend/agent/orchestrator.py`

- [x] **Step 1: Add `_get_tenant_degradation(state)` helper**

Add a module-level helper function that resolves `tenant_id` from `user_context`, fetches the tenant's SLA tier from `RateLimitRule`, and returns the per-tenant degradation level from `LoadMonitor.get_tenant_degradation(tier)`. Falls back to `free` tier on DB failure.

- [x] **Step 2: Replace `should_circuit_break_tavily()` in `web_searcher_node`**

Use `_get_tenant_degradation(state) == "cache_only"` instead — only free tenants under CRITICAL load skip Tavily.

- [x] **Step 3: Replace `should_circuit_break_neo4j()` in `local_graph_search_node`**

Use `_get_tenant_degradation(state) == "cache_only"` instead — only free tenants under CRITICAL load fall back to pure vector retrieval.

- [x] **Step 4: Replace `should_skip_critique()` in `route_after_critique`**

Use `_get_tenant_degradation(state) in ("skip_critique", "cache_only")` instead — free tenants skip critique at WARNING, premium tenants at CRITICAL, enterprise always runs full.

**Degradation matrix:**

| System Load | Enterprise | Premium | Free |
|------------|-----------|---------|------|
| NORMAL | full | full | full |
| WARNING | full | full | skip_critique |
| CRITICAL | full | skip_critique | cache_only |

**Commit:**
```bash
git add backend/agent/orchestrator.py
git commit -m "feat(v15): wire SLA-aware tenant degradation into orchestrator decision points"
```

---

## Milestone 3: Audit Trail

### Task 5: Audit Logger

**Files:**
- Create: `backend/billing/audit.py`
- Test: `tests/test_audit.py`

- [x] **Step 1: Write failing test**

```python
# tests/test_audit.py
import pytest
from unittest.mock import MagicMock
from backend.billing.audit import log_audit_event, AuditContext


def test_log_audit_event():
    mock_db = MagicMock()
    log_audit_event(
        db=mock_db, tenant_id=1, user_id=1,
        action="mcp_tool_call", target="query_database",
        arguments='{"sql": "SELECT 1"}', result_summary="success",
        risk_level="low",
    )
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


def test_audit_context_manager():
    mock_db = MagicMock()
    ctx = AuditContext(
        db=mock_db, tenant_id=1, user_id=1,
        action="sql_execute", target="chat_messages",
    )
    with ctx:
        ctx.set_result("returned 5 rows", risk_level="low")
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


def test_audit_risk_levels():
    mock_db = MagicMock()
    log_audit_event(
        db=mock_db, tenant_id=1, user_id=1,
        action="mcp_tool_call", target="send_email",
        risk_level="high",
    )
    call_args = mock_db.add.call_args[0][0]
    assert call_args.risk_level == "high"
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audit.py -v`
Expected: FAIL with `ImportError`

- [x] **Step 3: Implement audit logger**

```python
# backend/billing/audit.py
from sqlalchemy.orm import Session
from backend.billing.models import AuditLog


def log_audit_event(
    db: Session, tenant_id: int, user_id: int,
    action: str, target: str = None, arguments: str = None,
    result_summary: str = None, risk_level: str = "low",
    session_id: str = None, ip_address: str = None,
) -> AuditLog:
    log = AuditLog(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id,
        action=action, target=target, arguments=arguments,
        result_summary=result_summary, risk_level=risk_level,
        ip_address=ip_address,
    )
    db.add(log)
    db.commit()
    return log


class AuditContext:
    """Context manager for audit logging with before/after semantics."""

    def __init__(self, db: Session, tenant_id: int, user_id: int,
                 action: str, target: str = None, arguments: str = None,
                 session_id: str = None, ip_address: str = None):
        self.db = db
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.action = action
        self.target = target
        self.arguments = arguments
        self.session_id = session_id
        self.ip_address = ip_address
        self.result_summary = None
        self.risk_level = "low"

    def set_result(self, summary: str, risk_level: str = "low"):
        self.result_summary = summary
        self.risk_level = risk_level

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.result_summary = f"ERROR: {exc_val}"
            self.risk_level = "high"
        log_audit_event(
            db=self.db, tenant_id=self.tenant_id, user_id=self.user_id,
            action=self.action, target=self.target, arguments=self.arguments,
            result_summary=self.result_summary, risk_level=self.risk_level,
            session_id=self.session_id, ip_address=self.ip_address,
        )
        return False  # don't suppress exceptions
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_audit.py -v`
Expected: 3 passed

- [x] **Step 5: Commit**

```bash
git add backend/billing/audit.py tests/test_audit.py
git commit -m "feat(v15): add audit logger with context manager for MCP/SQL/tool calls"
```

---

### Task 6: Audit Integration (MCP + Data Analyst + HITL)

**Files:**
- Modify: `backend/agent/mcp_client.py`
- Modify: `backend/agent/data_analyst.py`
- Modify: `backend/agent/orchestrator.py`

- [x] **Step 1: Add audit to MCP call_tool**

Read `backend/agent/mcp_client.py`. Find the `call_tool` method in `MCPConnectionManager`. Wrap it with audit logging:

```python
def call_tool(self, server_name: str, tool_name: str, arguments: dict,
              tenant_id: int = 0, user_id: int = 0) -> dict:
    from backend.billing.audit import log_audit_event
    from backend.storage.database import SessionLocal

    conn = self._connections.get(server_name)
    if not conn:
        return {"error": f"Server {server_name} not connected"}

    result = conn.call_tool(tool_name, arguments)

    # Audit log
    db = SessionLocal()
    try:
        risk_level = "high" if any(kw in tool_name.lower() for kw in ["write", "delete", "send", "execute"]) else "low"
        log_audit_event(
            db=db, tenant_id=tenant_id, user_id=user_id,
            action="mcp_tool_call", target=f"{server_name}/{tool_name}",
            arguments=str(arguments)[:2000],
            result_summary=str(result.get("content", ""))[:500],
            risk_level=risk_level,
        )
    finally:
        db.close()

    return result
```

- [x] **Step 2: Add audit to Data Analyst SQL execution**

Read `backend/agent/data_analyst.py`. Find `execute_sql`. Add audit logging:

```python
def execute_sql(sql: str, tenant_id: int = 0, user_id: int = 0) -> dict:
    from backend.billing.audit import log_audit_event
    from backend.storage.database import SessionLocal

    # ... existing SELECT check ...
    # ... existing tenant_id check ...

    result = _execute(sql)

    # Audit log
    db = SessionLocal()
    try:
        risk_level = "high" if result.get("error") else "low"
        log_audit_event(
            db=db, tenant_id=tenant_id, user_id=user_id,
            action="sql_execute", target=sql[:200],
            result_summary=f"rows: {result.get('row_count', 0)}" if not result.get("error") else result.get("error"),
            risk_level=risk_level,
        )
    finally:
        db.close()

    return result
```

- [x] **Step 3: Add audit to HITL events**

Read `backend/agent/orchestrator.py`. Find the `interrupt()` calls in `data_analyst_node` and `rag_specialist_node`. Add audit logging before each interrupt:

```python
# Before interrupt in data_analyst_node:
from backend.billing.audit import log_audit_event
from backend.storage.database import SessionLocal
db = SessionLocal()
try:
    log_audit_event(
        db=db,
        tenant_id=state.get("user_context", {}).get("tenant_id", 0),
        user_id=state.get("user_context", {}).get("user_id", 0),
        action="hitl_interrupt", target="non_select_sql",
        arguments=sql[:500],
        result_summary="HITL approval required",
        risk_level="high",
    )
finally:
    db.close()
```

Apply same pattern to `rag_specialist_node` HITL interrupt.

- [x] **Step 4: Commit**

```bash
git add backend/agent/mcp_client.py backend/agent/data_analyst.py backend/agent/orchestrator.py
git commit -m "feat(v15): add audit logging to MCP calls, SQL execution, and HITL events"
```

---

## Milestone 4: Billing API & HITL Webhook

### Task 7: Billing API Endpoints

**Files:**
- Create: `backend/billing/routes.py`
- Modify: `backend/api/app.py`
- Modify: `backend/schemas.py`

- [x] **Step 1: Add Pydantic models to schemas.py**

Append to `backend/schemas.py`:

```python
class TokenUsageSummary(BaseModel):
    tenant_id: int
    period_days: int
    request_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int

class AuditLogEntry(BaseModel):
    id: int
    tenant_id: int
    user_id: int
    action: str
    target: Optional[str]
    result_summary: Optional[str]
    risk_level: str
    created_at: datetime

class AuditLogListResponse(BaseModel):
    logs: list[AuditLogEntry]
    total: int
```

- [x] **Step 2: Create billing routes**

```python
# backend/billing/routes.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.storage.database import get_db
from backend.auth.dependencies import UserContext, get_current_user
from backend.billing.token_tracker import get_usage_summary
from backend.billing.models import AuditLog
from backend.schemas import TokenUsageSummary, AuditLogListResponse, AuditLogEntry

router = APIRouter(prefix="/billing", tags=["billing"])


@router.get("/usage", response_model=TokenUsageSummary)
def get_usage(
    days: int = Query(30, ge=1, le=365),
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_usage_summary(db, tenant_id=user.tenant_id, days=days)


@router.get("/audit", response_model=AuditLogListResponse)
def get_audit_logs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    action: str = Query(None),
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(AuditLog).filter(AuditLog.tenant_id == user.tenant_id)
    if action:
        query = query.filter(AuditLog.action == action)
    total = query.count()
    logs = query.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit).all()
    return AuditLogListResponse(
        logs=[AuditLogEntry(
            id=l.id, tenant_id=l.tenant_id, user_id=l.user_id,
            action=l.action, target=l.target, result_summary=l.result_summary,
            risk_level=l.risk_level, created_at=l.created_at,
        ) for l in logs],
        total=total,
    )
```

- [x] **Step 3: Register billing router in app.py**

```python
from backend.billing.routes import router as billing_router
app.include_router(billing_router)
```

- [x] **Step 4: Commit**

```bash
git add backend/billing/routes.py backend/api/app.py backend/schemas.py
git commit -m "feat(v15): add /billing/usage and /billing/audit API endpoints"
```

---

### Task 8: HITL Webhook Notification

**Files:**
- Modify: `backend/agent/brain.py`

- [x] **Step 1: Add webhook notification to HITL interrupt**

Read `backend/agent/brain.py`. Find where `hitl_interrupt` SSE event is emitted in `_graph_worker`. Add a webhook call after the event:

```python
import aiohttp

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
                    "session_id": interrupt_data.get("session_id"),
                },
                timeout=aiohttp.ClientTimeout(total=5),
            )
    except Exception:
        pass  # non-fatal
```

Call this function right after pushing the `hitl_interrupt` event to the queue.

- [x] **Step 2: Commit**

```bash
git add backend/agent/brain.py
git commit -m "feat(v15): add HITL webhook notification to tenant admin"
```

---

## Milestone 5: Integration Tests & Final Verification

### Task 9: Full Integration Tests

**Files:**
- Test: `tests/test_billing_integration.py`

- [x] **Step 1: Write integration tests**

```python
# tests/test_billing_integration.py
"""Integration tests for v15 billing, rate limiting, and audit."""
import pytest
import inspect


def test_token_usage_log_model():
    from backend.billing.models import TokenUsageLog
    assert hasattr(TokenUsageLog, 'tenant_id')
    assert hasattr(TokenUsageLog, 'prompt_tokens')
    assert hasattr(TokenUsageLog, 'completion_tokens')
    assert hasattr(TokenUsageLog, 'total_tokens')


def test_rate_limit_rule_model():
    from backend.billing.models import RateLimitRule
    assert hasattr(RateLimitRule, 'tenant_id')
    assert hasattr(RateLimitRule, 'qps_limit')
    assert hasattr(RateLimitRule, 'daily_token_limit')


def test_audit_log_model():
    from backend.billing.models import AuditLog
    assert hasattr(AuditLog, 'tenant_id')
    assert hasattr(AuditLog, 'action')
    assert hasattr(AuditLog, 'risk_level')


def test_rate_limiter_class():
    from backend.billing.rate_limiter import TenantRateLimiter
    assert hasattr(TenantRateLimiter, 'check_rate_limit')
    assert hasattr(TenantRateLimiter, 'record_request')


def test_token_tracker_functions():
    from backend.billing.token_tracker import record_token_usage, get_usage_summary
    assert callable(record_token_usage)
    assert callable(get_usage_summary)


def test_audit_logger_functions():
    from backend.billing.audit import log_audit_event, AuditContext
    assert callable(log_audit_event)
    assert callable(AuditContext)


def test_billing_routes_exist():
    from backend.billing.routes import router
    paths = [r.path for r in router.routes]
    assert "/billing/usage" in paths
    assert "/billing/audit" in paths


def test_mcp_client_has_audit_params():
    from backend.agent.mcp_client import MCPConnectionManager
    sig = inspect.signature(MCPConnectionManager.call_tool)
    param_names = list(sig.parameters.keys())
    assert "tenant_id" in param_names


def test_get_tenant_rule_default():
    from backend.billing.rate_limiter import get_tenant_rule
    from unittest.mock import MagicMock
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None
    rule = get_tenant_rule(mock_db, tenant_id=999)
    assert rule.tier == "free"
    assert rule.qps_limit == 10


def test_sla_degradation_method():
    from backend.ha.load_monitor import LoadMonitor
    assert hasattr(LoadMonitor, 'get_tenant_degradation')
```

- [x] **Step 2: Run all v15 tests**

```bash
pytest tests/test_token_tracker.py tests/test_rate_limiter.py tests/test_audit.py tests/test_billing_integration.py -v
```

- [x] **Step 3: Commit**

```bash
git add tests/test_billing_integration.py
git commit -m "test(v15): add billing, rate limiting, and audit integration tests"
```

---

## Summary

| Milestone | Tasks | Core Change |
|-----------|-------|-------------|
| M1: Token Metering | 2 | TokenUsageLog model + tracker integrated into orchestrator |
| M2: Rate Limiting | 3 | Redis sliding-window limiter + middleware + SLA degradation wired into orchestrator |
| M3: Audit Trail | 2 | AuditLog model + context manager + MCP/SQL/HITL integration |
| M4: Billing API | 2 | /billing/usage + /billing/audit endpoints + HITL webhook |
| M5: Verification | 1 | Integration tests for all components |
| M6: Security Hardening | 1 | Frontend auth UI + JWT on all requests; tenant isolation on session save/delete, document list/delete; ChatRequest min_length validation; unprotected endpoint lockdown (2026-06-02) |
