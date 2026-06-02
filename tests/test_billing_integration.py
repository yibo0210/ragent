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
    try:
        from backend.agent.mcp_client import MCPConnectionManager
    except ImportError:
        pytest.skip("mcp module not installed")
    sig = inspect.signature(MCPConnectionManager.call_tool)
    param_names = list(sig.parameters.keys())
    assert "tenant_id" in param_names


def test_data_analyst_has_audit_params():
    from backend.agent.data_analyst import execute_sql
    sig = inspect.signature(execute_sql)
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


def test_billing_pydantic_models():
    from backend.schemas import TokenUsageSummary, AuditLogEntry, AuditLogListResponse
    assert 'tenant_id' in TokenUsageSummary.model_fields
    assert 'action' in AuditLogEntry.model_fields
    assert 'logs' in AuditLogListResponse.model_fields


def test_hitl_webhook_function_exists():
    from backend.agent import brain
    assert hasattr(brain, '_notify_hitl_webhook')
