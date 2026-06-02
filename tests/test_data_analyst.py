"""Test Data Analyst SQL execution safety."""
import pytest
from unittest.mock import MagicMock, patch


class TestExecuteSql:
    def test_rejects_non_select(self):
        from backend.agent.data_analyst import execute_sql
        result = execute_sql("DELETE FROM users WHERE 1=1")
        assert result["error"] == "non_select"

    def test_rejects_drop_table(self):
        from backend.agent.data_analyst import execute_sql
        result = execute_sql("DROP TABLE chat_sessions")
        assert result["error"] == "non_select"

    def test_rejects_insert(self):
        from backend.agent.data_analyst import execute_sql
        result = execute_sql("INSERT INTO users VALUES (1, 'test')")
        assert result["error"] == "non_select"

    def test_accepts_select(self):
        from backend.agent.data_analyst import execute_sql
        result = execute_sql("SELECT 1")
        assert result.get("error") is None or result.get("error") == "execution_failed"
        # "execution_failed" is OK here — it means SELECT passed the guard but the
        # test DB doesn't have the table. The key is "non_select" was NOT returned.

    def test_rejects_select_on_tenant_table_without_filter(self):
        from backend.agent.data_analyst import execute_sql
        result = execute_sql("SELECT * FROM chat_sessions")
        # Must be rejected due to missing tenant_id filter
        assert "SECURITY" in result.get("message", "")

    def test_cte_select_not_bypassed(self):
        from backend.agent.data_analyst import execute_sql
        result = execute_sql("WITH cte AS (DELETE FROM users RETURNING *) SELECT * FROM cte")
        assert result["error"] == "non_select"

    def test_semicolon_injection(self):
        from backend.agent.data_analyst import execute_sql
        result = execute_sql("SELECT 1; DROP TABLE users;")
        assert result["error"] == "non_select"


class TestSlaDegradation:
    def test_degradation_helper_exists(self):
        from backend.agent.orchestrator import _get_tenant_degradation
        assert callable(_get_tenant_degradation)

    def test_degradation_levels(self):
        from backend.ha.load_monitor import LoadMonitor, SystemState
        monitor = LoadMonitor(redis_client=MagicMock(), window=10)
        # Mock the state
        monitor.get_state = MagicMock(return_value=SystemState.CRITICAL)
        assert monitor.get_tenant_degradation("enterprise") == "full"
        assert monitor.get_tenant_degradation("premium") == "skip_critique"
        assert monitor.get_tenant_degradation("free") == "cache_only"

        monitor.get_state = MagicMock(return_value=SystemState.WARNING)
        assert monitor.get_tenant_degradation("free") == "skip_critique"
        assert monitor.get_tenant_degradation("enterprise") == "full"
