"""Tests for global load monitor with Redis-backed QPS tracking."""
import time
from unittest.mock import MagicMock, patch

from backend.ha.load_monitor import LoadMonitor, SystemState, get_load_monitor


class TestLoadMonitor:
    def _make_monitor(self, window=10, warning_qps=50, critical_qps=100):
        """Create a LoadMonitor with a mocked Redis client."""
        mock_redis = MagicMock()
        monitor = LoadMonitor(
            redis_client=mock_redis,
            window=window,
            warning_qps=warning_qps,
            critical_qps=critical_qps,
        )
        return monitor, mock_redis

    def test_initial_state_normal(self):
        """新创建的 LoadMonitor 初始状态应为 NORMAL。"""
        monitor, mock_redis = self._make_monitor()
        # Redis 返回 0（无流量）
        mock_redis.mget.return_value = ["0"] * 10
        assert monitor.get_state() == SystemState.NORMAL
        assert monitor._state == SystemState.NORMAL

    def test_state_transitions(self):
        """QPS 变化应触发状态从 NORMAL → WARNING → CRITICAL 转换。"""
        monitor, mock_redis = self._make_monitor(window=2, warning_qps=50, critical_qps=100)

        # NORMAL: QPS < 50 → sum/2 < 50 → sum < 100
        mock_redis.mget.return_value = ["20", "20"]  # QPS=20
        monitor._state = None  # clear cache
        assert monitor.get_state() == SystemState.NORMAL

        # WARNING: QPS >= 50 → sum/2 >= 50 → sum >= 100
        mock_redis.mget.return_value = ["50", "50"]  # QPS=50
        monitor._state = None
        assert monitor.get_state() == SystemState.WARNING

        # CRITICAL: QPS >= 100 → sum/2 >= 100 → sum >= 200
        mock_redis.mget.return_value = ["100", "100"]  # QPS=100
        monitor._state = None
        assert monitor.get_state() == SystemState.CRITICAL

    def test_should_skip_critique(self):
        """WARNING 及以上状态应跳过 Critique。"""
        monitor, mock_redis = self._make_monitor(window=2, warning_qps=50, critical_qps=100)

        # NORMAL → 不跳过
        mock_redis.mget.return_value = ["10", "10"]  # QPS=10
        monitor._state = None
        assert monitor.should_skip_critique() is False

        # WARNING → 跳过
        mock_redis.mget.return_value = ["50", "50"]  # QPS=50
        monitor._state = None
        assert monitor.should_skip_critique() is True

        # CRITICAL → 跳过
        mock_redis.mget.return_value = ["100", "100"]  # QPS=100
        monitor._state = None
        assert monitor.should_skip_critique() is True

    def test_should_circuit_break_neo4j(self):
        """只有 CRITICAL 状态应熔断 Neo4j。"""
        monitor, mock_redis = self._make_monitor(window=2, warning_qps=50, critical_qps=100)

        # NORMAL → 不熔断
        mock_redis.mget.return_value = ["10", "10"]  # QPS=10
        monitor._state = None
        assert monitor.should_circuit_break_neo4j() is False

        # WARNING → 不熔断
        mock_redis.mget.return_value = ["50", "50"]  # QPS=50
        monitor._state = None
        assert monitor.should_circuit_break_neo4j() is False

        # CRITICAL → 熔断
        mock_redis.mget.return_value = ["100", "100"]  # QPS=100
        monitor._state = None
        assert monitor.should_circuit_break_neo4j() is True

    def test_should_circuit_break_tavily(self):
        """只有 CRITICAL 状态应熔断 Tavily。"""
        monitor, mock_redis = self._make_monitor(window=2, warning_qps=50, critical_qps=100)

        # NORMAL → 不熔断
        mock_redis.mget.return_value = ["10", "10"]  # QPS=10
        monitor._state = None
        assert monitor.should_circuit_break_tavily() is False

        # WARNING → 不熔断
        mock_redis.mget.return_value = ["50", "50"]  # QPS=50
        monitor._state = None
        assert monitor.should_circuit_break_tavily() is False

        # CRITICAL → 熔断
        mock_redis.mget.return_value = ["100", "100"]  # QPS=100
        monitor._state = None
        assert monitor.should_circuit_break_tavily() is True

    def test_record_request(self):
        """record_request 应调用 Redis pipeline 执行 INCR + EXPIRE。"""
        monitor, mock_redis = self._make_monitor()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        monitor.record_request()

        mock_pipe.incr.assert_called_once()
        mock_pipe.expire.assert_called_once()
        mock_pipe.execute.assert_called_once()

    def test_record_request_redis_failure_silent(self):
        """Redis 不可用时 record_request 不应抛出异常。"""
        monitor, mock_redis = self._make_monitor()
        mock_redis.pipeline.side_effect = ConnectionError("Redis down")

        # 不应抛出异常
        monitor.record_request()

    def test_get_state_cache(self):
        """get_state 应在 1 秒内返回缓存状态，不重复查询 Redis。"""
        monitor, mock_redis = self._make_monitor(window=2, warning_qps=50, critical_qps=100)

        mock_redis.mget.return_value = ["10", "10"]
        state1 = monitor.get_state()
        state2 = monitor.get_state()

        # mget 只调用一次（第二次走缓存）
        assert mock_redis.mget.call_count == 1
        assert state1 == state2 == SystemState.NORMAL

    def test_get_state_cache_expired(self):
        """缓存过期后 get_state 应重新查询 Redis。"""
        monitor, mock_redis = self._make_monitor(window=2, warning_qps=50, critical_qps=100)

        mock_redis.mget.return_value = ["10", "10"]
        monitor.get_state()
        assert mock_redis.mget.call_count == 1

        # 模拟缓存过期
        monitor._state_ts = time.time() - 2
        monitor.get_state()
        assert mock_redis.mget.call_count == 2

    def test_evaluate_state_qps_calculation(self):
        """QPS 应等于窗口内所有计数之和除以窗口大小。"""
        monitor, mock_redis = self._make_monitor(window=5, warning_qps=50, critical_qps=100)

        # 窗口内各秒计数: [10, 20, 30, 40, 50] → sum=150, QPS=150/5=30 < 50 → NORMAL
        mock_redis.mget.return_value = ["10", "20", "30", "40", "50"]
        state = monitor.evaluate_state()
        assert state == SystemState.NORMAL

    def test_evaluate_state_critical(self):
        """高 QPS 应触发 CRITICAL 状态。"""
        monitor, mock_redis = self._make_monitor(window=5, warning_qps=50, critical_qps=100)

        # sum=500, avg=100 → CRITICAL
        mock_redis.mget.return_value = ["100", "100", "100", "100", "100"]
        state = monitor.evaluate_state()
        assert state == SystemState.CRITICAL

    def test_get_stats(self):
        """get_stats 应返回包含 state、qps、config 的字典。"""
        monitor, mock_redis = self._make_monitor(window=2, warning_qps=50, critical_qps=100)

        mock_redis.mget.return_value = ["20", "20"]
        stats = monitor.get_stats()

        assert "state" in stats
        assert "qps" in stats
        assert "warning_qps" in stats
        assert "critical_qps" in stats
        assert "window" in stats
        assert stats["state"] == "normal"
        assert stats["warning_qps"] == 50
        assert stats["critical_qps"] == 100

    def test_redis_read_failure_returns_zero(self):
        """Redis 读取失败时 QPS 应返回 0.0。"""
        monitor, mock_redis = self._make_monitor()
        mock_redis.mget.side_effect = ConnectionError("Redis down")

        state = monitor.evaluate_state()
        assert state == SystemState.NORMAL  # QPS=0 → NORMAL

    def test_mget_partial_none(self):
        """Redis mget 返回部分 None 时应正确计算 QPS。"""
        monitor, mock_redis = self._make_monitor(window=4, warning_qps=50, critical_qps=100)

        mock_redis.mget.return_value = ["10", None, "20", None]
        state = monitor.evaluate_state()
        # sum=30, avg=30/4=7.5 → NORMAL
        assert state == SystemState.NORMAL


class TestGetLoadMonitor:
    @patch.dict("os.environ", {"REDIS_URL": "redis://fake:6379/0"})
    @patch("backend.ha.load_monitor.redis.Redis")
    def test_singleton(self, mock_redis_cls):
        """get_load_monitor 应返回单例。"""
        import backend.ha.load_monitor as lm_mod
        lm_mod._load_monitor = None

        m1 = get_load_monitor()
        m2 = get_load_monitor()
        assert m1 is m2

    @patch.dict("os.environ", {
        "REDIS_URL": "redis://fake:6379/0",
        "LOAD_MONITOR_WINDOW": "30",
        "LOAD_WARNING_QPS": "80",
        "LOAD_CRITICAL_QPS": "200",
    })
    @patch("backend.ha.load_monitor.redis.Redis")
    def test_env_config(self, mock_redis_cls):
        """get_load_monitor 应从环境变量读取配置。"""
        import backend.ha.load_monitor as lm_mod
        lm_mod._load_monitor = None

        monitor = get_load_monitor()
        assert monitor.window == 30
        assert monitor.warning_qps == 80
        assert monitor.critical_qps == 200


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
