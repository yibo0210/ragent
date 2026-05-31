"""arq task queue configuration for async document ingestion."""
from arq.connections import RedisSettings

REDIS_URL = "redis://localhost:6379/1"


def get_redis_settings():
    return RedisSettings.from_dsn(REDIS_URL)
