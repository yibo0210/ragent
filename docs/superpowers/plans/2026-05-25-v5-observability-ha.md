# Ragent AI v5.0 — 全链路可观测性与高可用架构 升级计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 OpenTelemetry + Jaeger + Prometheus + Grafana 三维监控体系，引入熔断器、检索降级、指数退避重试机制，让系统从"算法实验"蜕变为"工业级中台"。

**Architecture:** 分两条并行轨道 — 可观测性（Phase 1-3）与高可用（Phase 4-6），最后 Docker Compose 统一编排（Phase 7）和 CI 更新（Phase 8）。

**Tech Stack:** OpenTelemetry SDK · Jaeger · Prometheus · Grafana · structlog · pybreaker · tenacity · Docker Compose

---

## 文件结构概览

```
新增文件 (10):
  backend/observability/__init__.py          # 可观测性模块入口
  backend/observability/tracing.py           # OTel 初始化 + Span 工具函数
  backend/observability/metrics.py           # Prometheus 指标注册
  backend/observability/logging.py           # Structlog 结构化日志配置
  backend/ha/__init__.py                     # 高可用模块入口
  backend/ha/circuit_breaker.py              # 熔断器装饰器/状态机
  backend/ha/retry.py                        # 指数退避重试装饰器
  backend/ha/degradation.py                  # 检索降级策略（Neo4j timeout → fallback）
  prometheus.yml                             # Prometheus 抓取配置
  grafana-dashboards/ragent-overview.json    # Grafana 预置仪表盘

修改文件 (12):
  backend/api/app.py                         # +OTel middleware, +/metrics endpoint
  backend/agent/orchestrator.py              # +自定义 Span 埋点（每个 Agent 节点）
  backend/agent/brain.py                     # +Trace context 注入, +structlog 日志
  backend/agent/tools.py                     # +LLM 调用 Span
  backend/milvus/client.py                   # +检索 Span, +timeout retry
  backend/storage/graph_client.py            # +Neo4j Span, +query timeout
  backend/rag/utils.py                       # +LLM invoke Span
  backend/storage/cache.py                   # +Redis 操作 Span
  docker-compose.yml                         # +Jaeger, Prometheus, Grafana 服务
  docker-compose.ci.yml                      # +监控服务
  pyproject.toml                             # +opentelemetry, prometheus-client, structlog, pybreaker, tenacity
  .env.example                               # +监控相关环境变量
```

---

## Phase 1: OpenTelemetry + Jaeger 分布式追踪

### Task 1.1: OTel SDK 初始化模块

**Files:**
- Create: `backend/observability/__init__.py`
- Create: `backend/observability/tracing.py`

- [ ] **Step 1: 创建 tracing.py — OTel 初始化 + Span 工具**

```python
# backend/observability/tracing.py
"""OpenTelemetry 分布式追踪初始化。"""
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

TRACING_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() != "false"
JAEGER_GRPC = os.getenv("JAEGER_GRPC_ENDPOINT", "http://localhost:4317")


def init_tracing(app=None) -> TracerProvider:
    if not TRACING_ENABLED:
        return None

    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=JAEGER_GRPC, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    if app:
        FastAPIInstrumentor.instrument_app(app)
    RedisInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument()

    return provider


def get_tracer(name: str = "ragent"):
    return trace.get_tracer(name)


def traced(name: str, attrs: dict = None):
    """装饰器：自动为函数创建 Span。"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(name, attributes=attrs or {}) as span:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR)
                    raise
        return wrapper
    return decorator


def current_span():
    return trace.get_current_span()
```

- [ ] **Step 2: 创建 __init__.py**

```python
# backend/observability/__init__.py
from .tracing import init_tracing, get_tracer, traced, current_span
from .metrics import init_metrics, Metrics
from .logging import init_logging, get_logger
```

- [ ] **Step 3: 验证模块导入**

```bash
uv run python -c "from backend.observability.tracing import init_tracing, get_tracer; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add backend/observability/
git commit -m "feat: add OpenTelemetry tracing init module"
```

---

### Task 1.2: FastAPI 中间件挂载 + Jaeger 导出

**Files:**
- Modify: `backend/api/app.py:21-46` (create_app)

- [ ] **Step 1: 在 create_app 中挂载 OTel 和 /metrics**

```python
# backend/api/app.py — 在 CORSMiddleware 之后、路由挂载之前插入
def create_app() -> FastAPI:
    app = FastAPI(title="Ragent AI API")

    @app.on_event("startup")
    async def _startup_init_db():
        init_db()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- v5.0 可观测性 ---
    from backend.observability import init_tracing, init_metrics, init_logging
    init_logging()
    init_tracing(app)
    init_metrics(app)
    # ---

    # ... 其余不变
```

- [ ] **Step 2: 添加 /metrics 端点（metrics.py 中处理）**

FastAPI 的 `/metrics` 端点由 prometheus-client 的 ASGI 中间件自动暴露，无需手动添加路由。

- [ ] **Step 3: 验证启动无报错**

```bash
uv run python -c "from backend.api.app import app; print('App created OK')"
```

Expected: 无 ImportError，FastAPIInstrumentor 正常加载。

- [ ] **Step 4: Commit**

```bash
git add backend/api/app.py
git commit -m "feat: mount OTel FastAPI middleware and metrics endpoint"
```

---

### Task 1.3: LangGraph Agent 节点埋点

**Files:**
- Modify: `backend/agent/orchestrator.py` (supervisor_node, rag_specialist_node, web_searcher_node, data_analyst_node, local_graph_search_node, global_graph_search_node, direct_answer_node)
- Modify: `backend/milvus/client.py` (hybrid_retrieve, dense_retrieve)
- Modify: `backend/storage/graph_client.py` (run_cypher, write_cypher)

- [ ] **Step 1: 每个 Agent 节点包裹 Span**

模式：在每个 Agent 节点的函数开头添加:

```python
from backend.observability import get_tracer

tracer = get_tracer("ragent.agent")

def rag_specialist_node(state: SupervisorState) -> dict:
    with tracer.start_as_current_span("agent.rag_specialist") as span:
        span.set_attribute("agent.name", "rag_specialist")
        span.set_attribute("user_query", state.get("user_query", "")[:200])
        # ... 原有逻辑
```

同样的模式应用到其余 6 个 Agent 节点。

- [ ] **Step 2: Milvus 检索添加 Span**

在 `backend/milvus/client.py` 的 `hybrid_retrieve` 和 `dense_retrieve` 方法中:

```python
from backend.observability import get_tracer

def hybrid_retrieve(self, dense_embedding, sparse_embedding, top_k=5, filter_expr=""):
    tracer = get_tracer("ragent.milvus")
    with tracer.start_as_current_span("milvus.hybrid_retrieve") as span:
        span.set_attribute("top_k", top_k)
        t0 = time.time()
        result = self._do_hybrid_retrieve(...)  # 原逻辑
        span.set_attribute("duration_ms", (time.time() - t0) * 1000)
        span.set_attribute("result_count", len(result) if result else 0)
        return result
```

- [ ] **Step 3: Neo4j Cypher 查询添加 Span**

在 `backend/storage/graph_client.py` 的 `run_cypher` 中:

```python
from backend.observability import get_tracer

def run_cypher(query: str, params: dict = None) -> list[dict]:
    tracer = get_tracer("ragent.neo4j")
    with tracer.start_as_current_span("neo4j.run_cypher") as span:
        span.set_attribute("cypher.query", query[:200])
        t0 = time.time()
        # ... 原有逻辑
        span.set_attribute("duration_ms", (time.time() - t0) * 1000)
        return result
```

- [ ] **Step 4: 验证 Span 不阻断业务**

```bash
# 即使 Jaeger 不可达也不应崩溃
uv run python -c "
from backend.observability import get_tracer
tracer = get_tracer('test')
with tracer.start_as_current_span('test'):
    print('Span OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add backend/agent/orchestrator.py backend/milvus/client.py backend/storage/graph_client.py
git commit -m "feat: add OTel spans to LangGraph nodes, Milvus, and Neo4j queries"
```

---

## Phase 2: Prometheus + Grafana 指标监控

### Task 2.1: Prometheus 指标定义 + /metrics 端点

**Files:**
- Create: `backend/observability/metrics.py`

- [ ] **Step 1: 注册核心指标**

```python
# backend/observability/metrics.py
"""Prometheus 指标定义与暴露。"""
import os
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST

METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() != "false"

llm_token_total = Counter(
    "llm_token_usage_total", "LLM Token 消耗", ["model", "direction"]
)
agent_routing = Counter(
    "agent_routing_count", "Supervisor 路由到各 Agent 的次数", ["agent"]
)
vector_search_latency = Histogram(
    "vector_search_latency_seconds", "向量检索延迟",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
graph_query_latency = Histogram(
    "graph_query_latency_seconds", "图查询延迟",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
llm_call_latency = Histogram(
    "llm_call_latency_seconds", "LLM 调用延迟",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
active_requests = Gauge(
    "active_requests", "当前活跃请求数"
)
circuit_breaker_state = Gauge(
    "circuit_breaker_state", "熔断器状态 (0=CLOSED, 1=OPEN, 2=HALF_OPEN)", ["service"]
)


class Metrics:
    @staticmethod
    def record_llm_tokens(model: str, input_tokens: int, output_tokens: int):
        llm_token_total.labels(model=model, direction="input").inc(input_tokens)
        llm_token_total.labels(model=model, direction="output").inc(output_tokens)

    @staticmethod
    def record_routing(agent: str):
        agent_routing.labels(agent=agent).inc()

    @staticmethod
    def record_vector_search(duration_s: float):
        vector_search_latency.observe(duration_s)

    @staticmethod
    def record_graph_query(duration_s: float):
        graph_query_latency.observe(duration_s)

    @staticmethod
    def record_llm_call(duration_s: float):
        llm_call_latency.observe(duration_s)

    @staticmethod
    def set_circuit_breaker(service: str, state: int):
        circuit_breaker_state.labels(service=service).set(state)


def init_metrics(app):
    if not METRICS_ENABLED:
        return

    from fastapi import Request
    from fastapi.responses import Response

    @app.get("/metrics")
    async def _metrics():
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.middleware("http")
    async def _track_active_requests(request: Request, call_next):
        active_requests.inc()
        try:
            return await call_next(request)
        finally:
            active_requests.dec()
```

- [ ] **Step 2: 验证 /metrics 端点**

```bash
uv run python -c "
from backend.observability.metrics import init_metrics
print('Metrics module OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add backend/observability/metrics.py
git commit -m "feat: add Prometheus metrics — tokens, routing, latency, circuit-breaker"
```

---

## Phase 3: 结构化 JSON 日志

### Task 3.1: Structlog 全局配置

**Files:**
- Create: `backend/observability/logging.py`

- [ ] **Step 1: 配置 structlog**

```python
# backend/observability/logging.py
"""Structlog 结构化 JSON 日志配置。"""
import os
import structlog
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # json | console


def init_logging():
    if LOG_FORMAT == "json":
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    logging.basicConfig(level=LOG_LEVEL, format="%(message)s")


def get_logger(name: str = "ragent"):
    return structlog.get_logger(name)
```

- [ ] **Step 2: 在 Agent 关键节点替换日志**

在 `backend/agent/brain.py` 顶部替换日志模式。在 `backend/agent/orchestrator.py` 的 supervisor_node 路由决策中使用结构化日志:

```python
from backend.observability import get_logger

log = get_logger("ragent.agent")

# 在 supervisor_node 中:
log.info("routing_decision", routes=routes, reason=reason, query=user_query[:100])

# 在 HITL 中断处:
log.warning("hitl_interrupt", session_id=session_id, scenario="rag_low_confidence")
```

- [ ] **Step 3: 验证日志输出**

```bash
uv run python -c "
from backend.observability.logging import init_logging, get_logger
init_logging()
log = get_logger('test')
log.info('test_log', key='value')
"
```

Expected: 输出一行 JSON 格式日志。

- [ ] **Step 4: Commit**

```bash
git add backend/observability/logging.py
git commit -m "feat: add structlog JSON logging configuration"
```

---

## Phase 4: 熔断器 (Circuit Breaker)

### Task 4.1: 通用熔断器装饰器

**Files:**
- Create: `backend/ha/__init__.py`
- Create: `backend/ha/circuit_breaker.py`

- [ ] **Step 1: 实现熔断器状态机**

```python
# backend/ha/circuit_breaker.py
"""熔断器模式：外部 API 调用保护。"""
import time
import functools
from enum import Enum
from backend.observability import get_logger, Metrics

log = get_logger("ragent.ha")


class State(Enum):
    CLOSED = 0      # 正常
    OPEN = 1        # 熔断
    HALF_OPEN = 2   # 试探


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3,
                 recovery_timeout: float = 60.0, half_open_max: int = 1):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self.state = State.CLOSED
        self.failures = 0
        self.last_failure_time = 0.0
        self._half_open_count = 0

    def _transition(self, new_state: State):
        self.state = new_state
        Metrics.set_circuit_breaker(self.name, new_state.value)
        log.warning("circuit_breaker_transition", service=self.name,
                     state=new_state.name)

    def call(self, func, *args, **kwargs):
        if self.state == State.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self._transition(State.HALF_OPEN)
                self._half_open_count = 0
            else:
                raise CircuitBreakerOpenError(self.name)

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        if self.state == State.HALF_OPEN:
            self._half_open_count += 1
            if self._half_open_count >= self.half_open_max:
                self._transition(State.CLOSED)
                self.failures = 0
        self.failures = max(0, self.failures - 1)

    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self._transition(State.OPEN)


class CircuitBreakerOpenError(Exception):
    def __init__(self, service: str):
        super().__init__(f"Circuit breaker OPEN for '{service}'")


# 全局熔断器实例
llm_breaker = CircuitBreaker("llm", failure_threshold=3, recovery_timeout=60.0)
tavily_breaker = CircuitBreaker("tavily", failure_threshold=3, recovery_timeout=60.0)


def with_circuit_breaker(breaker: CircuitBreaker, fallback=None):
    """装饰器：为函数包裹熔断保护。"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return breaker.call(func, *args, **kwargs)
            except CircuitBreakerOpenError:
                if fallback:
                    return fallback(*args, **kwargs)
                raise
        return wrapper
    return decorator
```

- [ ] **Step 2: 将熔断器应用到 LLM 调用**

在 `backend/agent/orchestrator.py` 的 `_get_worker_model()` 等方法中包裹 `.invoke()`:

```python
from backend.ha.circuit_breaker import with_circuit_breaker, llm_breaker

@with_circuit_breaker(llm_breaker, fallback=None)
def _safe_llm_invoke(model, messages):
    return model.invoke(messages)
```

- [ ] **Step 3: 验证熔断逻辑**

```bash
uv run python -c "
from backend.ha.circuit_breaker import CircuitBreaker
cb = CircuitBreaker('test', failure_threshold=2, recovery_timeout=1)
try: cb.call(lambda: 1/0)
except: pass
try: cb.call(lambda: 1/0)
except: pass
try: cb.call(lambda: 1)
except Exception as e: print(f'State: {cb.state.name}, Error: {e}')
"
```

Expected: 第三次调用抛出 `CircuitBreakerOpenError`。

- [ ] **Step 4: Commit**

```bash
git add backend/ha/
git commit -m "feat: add circuit breaker for LLM and Tavily external APIs"
```

---

## Phase 5: 检索降级策略

### Task 5.1: Neo4j 查询超时 + 自动降级

**Files:**
- Modify: `backend/storage/graph_client.py` (add timeout)
- Create: `backend/ha/degradation.py`

- [ ] **Step 1: Neo4j 驱动配置查询超时**

在 `backend/storage/graph_client.py` 的 `_get_driver()` 中追加超时配置:

```python
def _get_driver() -> Driver:
    global _neo4j_driver
    if _neo4j_driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        query_timeout = float(os.getenv("NEO4J_QUERY_TIMEOUT", "1.5"))
        _neo4j_driver = GraphDatabase.driver(
            uri, auth=(user, password),
            max_connection_lifetime=30,
            connection_acquisition_timeout=3,
        )
    return _neo4j_driver


def run_cypher(query: str, params: dict = None, timeout: float = None) -> list[dict]:
    """执行只读 Cypher 查询，支持超时。"""
    if timeout is None:
        timeout = float(os.getenv("NEO4J_QUERY_TIMEOUT", "1.5"))
    with _get_driver().session() as session:
        result = session.run(
            query, params or {},
            timeout=timeout,
        )
        return [dict(record) for record in result]
```

- [ ] **Step 2: 创建降级策略模块**

```python
# backend/ha/degradation.py
"""检索降级策略：Neo4j 超时 → 纯向量双路召回。"""
from backend.observability import get_logger

log = get_logger("ragent.ha")


def safe_graph_search(query: str, top_k: int = 5) -> dict:
    """包裹图搜索，超时时自动降级为纯向量检索。"""
    try:
        from backend.rag.graph_retriever import local_graph_search
        return local_graph_search(query, top_k)
    except Exception as e:
        log.warning("graph_search_degraded", query=query[:100], error=str(e))
        from backend.rag.utils import retrieve_documents
        result = retrieve_documents(query, top_k=top_k)
        return {
            **result,
            "mode": "degraded_dense_sparse",
            "degradation_reason": str(e)[:200],
        }
```

- [ ] **Step 3: 在 local_graph_search_node 中使用安全包裹**

```python
# backend/agent/orchestrator.py — local_graph_search_node
from backend.ha.degradation import safe_graph_search

# 替换原来的 local_graph_search(user_query, ...)
result = safe_graph_search(user_query)
```

- [ ] **Step 4: Commit**

```bash
git add backend/storage/graph_client.py backend/ha/degradation.py backend/agent/orchestrator.py
git commit -m "feat: add Neo4j query timeout and graceful degradation to Dense+Sparse"
```

---

## Phase 6: 指数退避重试

### Task 6.1: tenacity 重试装饰器

**Files:**
- Create: `backend/ha/retry.py`

- [ ] **Step 1: 实现通用重试装饰器**

```python
# backend/ha/retry.py
"""指数退避重试装饰器。"""
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from backend.observability import get_logger

log = get_logger("ragent.ha")


def with_retry(max_attempts: int = 3, min_wait: float = 1.0, max_wait: float = 10.0):
    """指数退避重试：适用于网络抖动场景。"""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=min_wait, max=max_wait),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
        before_sleep=lambda retry_state: log.warning(
            "retry_attempt",
            attempt=retry_state.attempt_number,
            exception=str(retry_state.outcome.exception())[:200],
        ),
    )
```

- [ ] **Step 2: 应用到 LLM 生成和数据库写入**

在 `backend/agent/orchestrator.py` 的 LLM invoke 调用上:

```python
from backend.ha.retry import with_retry

@with_retry(max_attempts=3)
def _invoke_with_retry(model, messages):
    return model.invoke(messages)
```

在 `backend/milvus/client.py` 的 `insert` 方法上:

```python
from backend.ha.retry import with_retry

@with_retry(max_attempts=3)
def insert(self, data):
    # ... 原有逻辑
```

- [ ] **Step 3: 验证重试逻辑**

```bash
uv run python -c "
from backend.ha.retry import with_retry

call_count = [0]
@with_retry(max_attempts=3, min_wait=0.1)
def flaky_func():
    call_count[0] += 1
    if call_count[0] < 3:
        raise ConnectionError('fail')
    return 'ok'

result = flaky_func()
print(f'Result: {result}, attempts: {call_count[0]}')
"
```

Expected: `Result: ok, attempts: 3`

- [ ] **Step 4: Commit**

```bash
git add backend/ha/retry.py backend/agent/orchestrator.py backend/milvus/client.py
git commit -m "feat: add exponential backoff retry for LLM and DB operations"
```

---

## Phase 7: Docker Compose 监控栈

### Task 7.1: Jaeger + Prometheus + Grafana 容器编排

**Files:**
- Modify: `docker-compose.yml`
- Create: `prometheus.yml`

- [ ] **Step 1: 追加监控服务到 docker-compose.yml**

```yaml
  # ===== v5.0 可观测性栈 =====
  jaeger:
    container_name: ragent-jaeger
    image: jaegertracing/all-in-one:1.58
    ports:
      - "16686:16686"   # Jaeger UI
      - "4317:4317"     # OTLP gRPC
      - "4318:4318"     # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - default

  prometheus:
    container_name: ragent-prometheus
    image: prom/prometheus:v2.52.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=15d'
    networks:
      - default

  grafana:
    container_name: ragent-grafana
    image: grafana/grafana:10.4.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_AUTH_ANONYMOUS_ENABLED=true
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - default
```

- [ ] **Step 2: 创建 prometheus.yml**

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ragent'
    static_configs:
      - targets: ['host.docker.internal:8000']
```

- [ ] **Step 3: 更新 volumes 声明**

在 `docker-compose.yml` 的 `volumes:` 顶部追加:

```yaml
  grafana_data:
```

- [ ] **Step 4: 验证服务启动**

```bash
docker compose up -d jaeger prometheus grafana
docker compose ps | grep -E "jaeger|prometheus|grafana"
```

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml prometheus.yml
git commit -m "feat: add Jaeger, Prometheus, Grafana to Docker Compose"
```

---

## Phase 8: 依赖更新 + CI 配置

### Task 8.1: pyproject.toml 追加依赖 + CI 验证

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: 追加监控 + HA 依赖**

```toml
"opentelemetry-api>=1.24.0",
"opentelemetry-sdk>=1.24.0",
"opentelemetry-instrumentation-fastapi>=0.45b0",
"opentelemetry-instrumentation-redis>=0.45b0",
"opentelemetry-instrumentation-sqlalchemy>=0.45b0",
"opentelemetry-exporter-otlp-proto-grpc>=1.24.0",
"prometheus-client>=0.20.0",
"structlog>=24.1.0",
"pybreaker>=0.8.0",
"tenacity>=8.3.0",
```

- [ ] **Step 2: 更新 .env.example**

```env
# ===== v5.0 Observability =====
OTEL_ENABLED=true
JAEGER_GRPC_ENDPOINT=http://localhost:4317
METRICS_ENABLED=true
LOG_LEVEL=INFO
LOG_FORMAT=json

# ===== v5.0 High Availability =====
NEO4J_QUERY_TIMEOUT=1.5
```

- [ ] **Step 3: 更新 CI 导入验证**

在 `.github/workflows/ci.yml` 的 import verification 步骤中追加:

```yaml
uv run python -c "
from backend.observability import init_tracing, init_metrics, init_logging
from backend.ha.circuit_breaker import CircuitBreaker, llm_breaker
from backend.ha.retry import with_retry
from backend.ha.degradation import safe_graph_search
print('v5.0 imports successful')
"
```

- [ ] **Step 4: 安装依赖并验证**

```bash
uv sync
uv run python -c "from backend.observability import init_tracing; from backend.ha.circuit_breaker import CircuitBreaker; print('All v5.0 deps OK')"
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .env.example .github/workflows/ci.yml
git commit -m "feat: add v5.0 observability and HA dependencies, CI updates"
```

---

## 验收标准

### Phase 1-3 — 可观测性
- [ ] Jaeger UI (localhost:16686) 可访问，能看到 Agent 节点 Span 链路
- [ ] Prometheus (localhost:9090) 能抓取到 `/metrics` 端点数据
- [ ] Grafana (localhost:3000) 添加 Prometheus 数据源后有数据
- [ ] 日志输出为 JSON 格式，包含 timestamp、level、event 字段
- [ ] OTel 服务不可达时业务不中断

### Phase 4-5 — 高可用
- [ ] LLM API 连续失败 3 次后熔断器打开，返回降级兜底回答
- [ ] Neo4j 查询超过 1.5 秒自动降级为纯向量检索
- [ ] 降级时记录 WARNING 日志

### Phase 6 — 重试
- [ ] 网络抖动时 LLM/DB 操作自动重试 3 次（指数退避）
- [ ] 重试信息记录在结构化日志中

### Phase 7-8 — 部署
- [ ] `docker compose up -d` 一键启动含监控栈的全部服务
- [ ] CI 流水线通过 v5.0 导入验证

---

## 执行顺序

```
Phase 1 (OTel Tracing) ──→ Phase 3 (Structlog) ──→ Phase 4 (Circuit Breaker)
                                                          │
Phase 2 (Prometheus) ─────────────────────────────────────┤
                                                          ▼
Phase 7 (Docker Compose) ←── Phase 6 (Retry) ←── Phase 5 (Degradation)
                                                          │
Phase 8 (CI) ─────────────────────────────────────────────┘
```

Phase 1、2、3 可并行（都是可观测性，互不依赖）。Phase 4、5、6 可并行（都是高可用，独立模块）。Phase 7 在所有基础设施就绪后统一编排。Phase 8 最后收尾。
