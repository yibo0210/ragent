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


from backend.billing.token_tracker import record_token_usage, get_usage_summary
from unittest.mock import MagicMock


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
