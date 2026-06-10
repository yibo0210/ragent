"""Centralized configuration with validation via Pydantic BaseSettings.

All env vars are validated at startup. No hardcoded fallback secrets.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # --- Model ---
    ark_api_key: str = ""
    model: str = "qwen-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedder: str = "text-embedding-v1"
    grade_model: str = "qwen-plus"
    max_tokens: int = 8192
    supervisor_model: str = ""
    model_turbo: str = "qwen-turbo"
    model_max: str = "qwen-max"

    # --- Database ---
    database_url: str = ""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_query_timeout: float = 1.5

    # --- Milvus ---
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_search_top_k: int = 20
    milvus_vector_dim: int = 1536

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- JWT ---
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # --- Rerank ---
    rerank_model: str = ""
    rerank_binding_host: str = ""
    rerank_api_key: str = ""
    rerank_top_k: int = 10

    # --- Tools ---
    tavily_api_key: str = ""
    amap_api_key: str = ""

    # --- RRF ---
    rrf_weight_dense: float = 0.4
    rrf_weight_sparse: float = 0.3
    rrf_weight_graph: float = 0.15
    rrf_weight_visual: float = 0.15

    # --- Cache ---
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 86400
    cache_enabled: bool = True

    # --- Observability ---
    otel_enabled: bool = False
    metrics_enabled: bool = True
    log_level: str = "INFO"
    log_format: str = "json"

    # --- HITL ---
    hitl_webhook_url: str = ""

    # --- MinIO ---
    minio_endpoint: str = "localhost:9000"
    minio_bucket: str = "ragent-media"
    minio_access_key: str = ""
    minio_secret_key: str = ""

    # --- Upload ---
    upload_max_size_mb: int = 50

    # --- CORS ---
    cors_origins: str = "*"

    # --- v19 Memory Graph ---
    memory_enabled: bool = False
    memory_extraction_model: str = ""
    memory_importance_threshold: float = 0.3

    # --- v20 Deep Research ---
    research_enabled: bool = True
    research_max_review_rounds: int = 3
    research_default_timeout_minutes: int = 30
    research_max_evidence_per_task: int = 20
    research_report_formats: str = "markdown,pdf"

    # --- Host/Port ---
    host: str = "0.0.0.0"
    port: int = 8000


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
