import pytest
from unittest.mock import MagicMock
from backend.billing.rate_limiter import TenantRateLimiter, get_tenant_rule


@pytest.fixture
def limiter():
    mock_redis = MagicMock()
    return TenantRateLimiter(mock_redis)


def test_rate_limiter_allows_under_limit(limiter):
    # window=10 by default, so mget returns 10 values; total=10 < 10*10=100
    limiter.redis.mget.return_value = [b"1"] * 10
    result = limiter.check_rate_limit(tenant_id=1, qps_limit=10)
    assert result["allowed"] is True


def test_rate_limiter_blocks_over_limit(limiter):
    # window=10, qps_limit=5, so limit=50; total=10*10=100 >= 50
    limiter.redis.mget.return_value = [b"10"] * 10
    result = limiter.check_rate_limit(tenant_id=1, qps_limit=5)
    assert result["allowed"] is False
    assert result["retry_after"] > 0


def test_rate_limiter_increments_counter(limiter):
    limiter.redis.pipeline.return_value = limiter.redis
    limiter.redis.execute.return_value = [None, None]
    limiter.record_request(tenant_id=1)
    limiter.redis.incr.assert_called()


def test_rate_limiter_tenant_specific(limiter):
    limiter.redis.mget.return_value = [b"0"]
    result = limiter.check_rate_limit(tenant_id=999, qps_limit=100)
    assert result["allowed"] is True


def test_get_tenant_rule_existing():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = MagicMock(
        qps_limit=20, daily_token_limit=500000, tier="standard"
    )
    rule = get_tenant_rule(mock_db, tenant_id=1)
    assert rule.qps_limit == 20
    assert rule.tier == "standard"


def test_get_tenant_rule_default():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None
    rule = get_tenant_rule(mock_db, tenant_id=999)
    assert rule.qps_limit == 10
    assert rule.tier == "free"
