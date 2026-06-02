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


def test_audit_context_on_exception():
    mock_db = MagicMock()
    with pytest.raises(ValueError):
        with AuditContext(db=mock_db, tenant_id=1, user_id=1, action="test") as ctx:
            raise ValueError("something broke")
    call_args = mock_db.add.call_args[0][0]
    assert call_args.risk_level == "high"
    assert "ERROR" in call_args.result_summary
