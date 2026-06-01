"""Prometheus 指标定义与暴露。"""
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY, CONTENT_TYPE_LATEST

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
active_requests = Gauge("active_requests", "当前活跃请求数")
circuit_breaker_state = Gauge(
    "circuit_breaker_state", "熔断器状态 (0=CLOSED, 1=OPEN, 2=HALF_OPEN)", ["service"]
)
system_state = Gauge(
    "system_load_state", "系统负载状态 (0=NORMAL, 1=WARNING, 2=CRITICAL)"
)
query_qps = Gauge(
    "query_qps", "当前查询 QPS"
)
profiler_distribution = Counter(
    "query_profiler_distribution", "Query Profiler 意图分布", ["level"]
)


class Metrics:
    @staticmethod
    def record_llm_tokens(model: str, input_tokens: int, output_tokens: int):
        if not METRICS_ENABLED:
            return
        llm_token_total.labels(model=model, direction="input").inc(input_tokens)
        llm_token_total.labels(model=model, direction="output").inc(output_tokens)

    @staticmethod
    def record_routing(agent: str):
        if not METRICS_ENABLED:
            return
        agent_routing.labels(agent=agent).inc()

    @staticmethod
    def record_vector_search(duration_s: float):
        if not METRICS_ENABLED:
            return
        vector_search_latency.observe(duration_s)

    @staticmethod
    def record_graph_query(duration_s: float):
        if not METRICS_ENABLED:
            return
        graph_query_latency.observe(duration_s)

    @staticmethod
    def record_llm_call(duration_s: float):
        if not METRICS_ENABLED:
            return
        llm_call_latency.observe(duration_s)

    @staticmethod
    def set_circuit_breaker(service: str, state: int):
        if not METRICS_ENABLED:
            return
        circuit_breaker_state.labels(service=service).set(state)

    @staticmethod
    def set_system_state(state_value: int):
        if not METRICS_ENABLED:
            return
        system_state.set(state_value)

    @staticmethod
    def set_qps(qps: float):
        if not METRICS_ENABLED:
            return
        query_qps.set(qps)

    @staticmethod
    def record_profiler_intent(level: str):
        if not METRICS_ENABLED:
            return
        profiler_distribution.labels(level=level).inc()


def init_metrics(app):
    if not METRICS_ENABLED:
        return

    from fastapi.responses import Response

    @app.get("/metrics")
    async def _metrics():
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.middleware("http")
    async def _track_active_requests(request, call_next):
        active_requests.inc()
        try:
            return await call_next(request)
        finally:
            active_requests.dec()
