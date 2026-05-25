"""Redis Singleflight：防缓存击穿。"""
import time
from backend.storage.cache import cache

SINGLEFLIGHT_TTL = 30


def with_singleflight(key_prefix: str, ttl: int = SINGLEFLIGHT_TTL):
    """装饰器：相同 key 的并发请求只穿透一次。"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            query = args[0] if args else kwargs.get("query", "")
            lock_key = f"singleflight:{key_prefix}:{hash(query) % 10000}"
            result_key = f"singleflight_result:{key_prefix}:{hash(query) % 10000}"

            acquired = cache._get_client().set(
                cache._key(lock_key), "1", nx=True, ex=ttl
            )
            if acquired:
                try:
                    result = func(*args, **kwargs)
                    cache.set_json(result_key, result, ttl=60)
                    return result
                finally:
                    cache.delete(lock_key)
            else:
                for _ in range(min(ttl * 2, 60)):
                    cached = cache.get_json(result_key)
                    if cached is not None:
                        return cached
                    time.sleep(0.5)
                return func(*args, **kwargs)
        return wrapper
    return decorator
