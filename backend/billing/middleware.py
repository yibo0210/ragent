from fastapi import Request
from fastapi.responses import JSONResponse
from backend.billing.rate_limiter import TenantRateLimiter, get_tenant_rule
from backend.storage.database import SessionLocal


def create_rate_limit_middleware(limiter: TenantRateLimiter):
    async def rate_limit_middleware(request: Request, call_next):
        path = request.url.path
        if not path.startswith("/chat") and not path.startswith("/documents"):
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return await call_next(request)

        try:
            from backend.auth.jwt_handler import decode_token
            token = auth_header.split(" ", 1)[1]
            payload = decode_token(token)
            tenant_id = payload.get("tenant_id", 0)
        except Exception:
            return await call_next(request)

        if not tenant_id:
            return await call_next(request)

        db = SessionLocal()
        try:
            rule = get_tenant_rule(db, tenant_id)
            result = limiter.check_rate_limit(tenant_id, rule.qps_limit)
        finally:
            db.close()

        if not result["allowed"]:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "retry_after": result["retry_after"]},
                headers={"Retry-After": str(result["retry_after"])},
            )

        limiter.record_request(tenant_id)
        response = await call_next(request)
        return response

    return rate_limit_middleware
