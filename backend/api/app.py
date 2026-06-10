"""Web 应用入口模块

基于 FastAPI 搭建后端服务，配置跨域、初始化数据库、挂载 API 路由和前端静态资源。
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .routes import router
from backend.storage.database import init_db

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

def create_app() -> FastAPI:
    app = FastAPI(title="Ragent AI API")

    @app.on_event("startup")
    async def _startup_init_db():
        init_db()
        # v12: 预热 Query Profiler 原型 Embedding（避免首次请求阻塞）
        try:
            from backend.agent.query_profiler import warmup
            warmup()
        except Exception as e:
            from backend.observability import get_logger
            get_logger("ragent.app").warning("query_profiler_warmup_failed", error=str(e))

    from backend.config import get_settings
    cors_origins = get_settings().cors_origins.split(",") if get_settings().cors_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # No-cache middleware for development
    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # --- v15 Rate-limit middleware ---
    try:
        from backend.billing.rate_limiter import TenantRateLimiter
        from backend.billing.middleware import create_rate_limit_middleware
        from backend.storage.cache import cache
        _limiter = TenantRateLimiter(cache._get_client())
        app.middleware("http")(create_rate_limit_middleware(_limiter))
    except Exception as e:
        from backend.observability import get_logger
        get_logger("ragent.app").warning("rate_limit_middleware_init_failed", error=str(e))

    app.include_router(router)

    # --- v14 Auth routes ---
    from backend.auth.routes import router as auth_router
    app.include_router(auth_router)

    # --- v15 Billing routes ---
    from backend.billing.routes import router as billing_router
    app.include_router(billing_router)

    # --- v16 Workflow routes ---
    try:
        from backend.workflow.routes import router as workflow_router
        app.include_router(workflow_router)
    except Exception as e:
        from backend.observability import get_logger
        get_logger("ragent.app").warning("workflow_routes_init_failed", error=str(e))

    # --- v20 Research routes ---
    try:
        from backend.research.routes import router as research_router
        app.include_router(research_router)
    except Exception as e:
        from backend.observability import get_logger
        get_logger("ragent.app").warning("research_routes_init_failed", error=str(e))

    # --- v5.0 可观测性 (must be after router + before static mount) ---
    from backend.observability import init_logging, init_tracing, init_metrics
    init_logging()
    init_tracing(app)
    init_metrics(app)
    # ---

    # serve frontend static files at root (must be last — overrides all paths)
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
