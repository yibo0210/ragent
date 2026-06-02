"""全局负载监控器：基于 Redis 滑动窗口的 QPS 监控与自适应降级。

QPS < 50  → NORMAL   — 全量链路
QPS 50-100 → WARNING  — 跳过 Critique/Replan
QPS > 100  → CRITICAL — 熔断 Neo4j 和 Tavily
"""
import os
import time
from enum import Enum
from typing import Optional

import redis

from backend.observability import get_logger, Metrics

log = get_logger("ragent.ha")


class SystemState(Enum):
    NORMAL = "normal"      # QPS < threshold, 全量链路
    WARNING = "warning"    # QPS >= warning, 跳过 Critique/Replan
    CRITICAL = "critical"  # QPS >= critical, 熔断 Neo4j 和 Tavily


class LoadMonitor:
    """基于 Redis 滑动窗口的全局 QPS 负载监控器。"""

    def __init__(
        self,
        redis_client: redis.Redis,
        window: int = 10,
        warning_qps: float = 50,
        critical_qps: float = 100,
    ):
        self._redis = redis_client
        self.window = window
        self.warning_qps = warning_qps
        self.critical_qps = critical_qps
        self._state: Optional[SystemState] = None
        self._state_ts: float = 0.0
        self._cache_ttl: float = 1.0  # 秒

    def record_request(self) -> None:
        """记录一次请求到当前秒的 Redis 计数器。INCR + EXPIRE pipeline。"""
        try:
            ts = int(time.time())
            key = f"ragent:load:qps:{ts}"
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.window)
            pipe.execute()
        except Exception as e:
            log.warning("load_monitor_redis_unavailable", error=str(e))

    def evaluate_state(self) -> SystemState:
        """根据当前 QPS 评估系统状态（强制查询 Redis）。"""
        qps = self._get_qps()
        state = self._classify(qps)
        self._state = state
        self._state_ts = time.time()
        log.info("load_monitor_evaluate", qps=qps, state=state.value)
        try:
            Metrics.set_circuit_breaker("load_monitor", {"normal": 0, "warning": 1, "critical": 2}.get(state.value, 0))
        except Exception as e:
            log.warning("load_monitor_metrics_failed", error=str(e))
        return state

    def get_state(self) -> SystemState:
        """获取系统状态（带 1 秒缓存）。"""
        now = time.time()
        if self._state is not None and now - self._state_ts < self._cache_ttl:
            return self._state
        return self.evaluate_state()

    def should_skip_critique(self) -> bool:
        """WARNING 及以上状态时跳过 Critique/Replan。"""
        return self.get_state() in (SystemState.WARNING, SystemState.CRITICAL)

    def should_circuit_break_neo4j(self) -> bool:
        """CRITICAL 状态时熔断 Neo4j 图查询。"""
        return self.get_state() == SystemState.CRITICAL

    def should_circuit_break_tavily(self) -> bool:
        """CRITICAL 状态时熔断 Tavily Web 搜索。"""
        return self.get_state() == SystemState.CRITICAL

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

    def get_stats(self) -> dict:
        """返回当前监控统计信息。"""
        state = self.get_state()
        qps = self._get_qps()
        return {
            "state": state.value,
            "qps": round(qps, 2),
            "window": self.window,
            "warning_qps": self.warning_qps,
            "critical_qps": self.critical_qps,
        }

    # -- internal --

    def _get_qps(self) -> float:
        """从 Redis 读取滑动窗口内的平均 QPS。"""
        try:
            now = int(time.time())
            keys = [f"ragent:load:qps:{now - i}" for i in range(self.window)]
            values = self._redis.mget(keys)
            total = sum(int(v) for v in values if v is not None)
            return total / self.window
        except Exception:
            return 0.0

    def _classify(self, qps: float) -> SystemState:
        if qps >= self.critical_qps:
            return SystemState.CRITICAL
        if qps >= self.warning_qps:
            return SystemState.WARNING
        return SystemState.NORMAL


# ---------- 模块级单例 ----------

_load_monitor: Optional[LoadMonitor] = None


def get_load_monitor() -> LoadMonitor:
    """获取全局负载监控器单例。"""
    global _load_monitor
    if _load_monitor is not None:
        return _load_monitor

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    window = int(os.getenv("LOAD_MONITOR_WINDOW", "10"))
    warning_qps = float(os.getenv("LOAD_WARNING_QPS", "50"))
    critical_qps = float(os.getenv("LOAD_CRITICAL_QPS", "100"))

    client = redis.Redis.from_url(redis_url, decode_responses=True)
    _load_monitor = LoadMonitor(
        redis_client=client,
        window=window,
        warning_qps=warning_qps,
        critical_qps=critical_qps,
    )
    log.info("load_monitor_initialized", window=window,
             warning_qps=warning_qps, critical_qps=critical_qps)
    return _load_monitor
