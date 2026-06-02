import time
from types import SimpleNamespace
from sqlalchemy.orm import Session
from backend.billing.models import RateLimitRule


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
            pass  # fail-open: Redis unavailable, skip rate limit counting

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


def get_tenant_rule(db: Session, tenant_id: int):
    rule = db.query(RateLimitRule).filter(RateLimitRule.tenant_id == tenant_id).first()
    if rule:
        return rule
    return SimpleNamespace(
        tenant_id=tenant_id, tier="free", qps_limit=10,
        daily_token_limit=100000, concurrent_limit=5,
    )
