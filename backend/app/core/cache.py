"""
VeriMind High-Performance Cache Manager.
Provides Redis caching with auto-falling back to In-Memory Dictionary for standalone operation.
"""
import json
import hashlib
from typing import Optional, Any
from backend.app.config import settings
from backend.app.core.logger import logger

class CacheManager:
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.using_redis = False
        self._init_redis()

    def _init_redis(self):
        try:
            import redis
            client = redis.Redis.from_url(settings.REDIS_URL, socket_timeout=1.0)
            client.ping()
            self.redis_client = client
            self.using_redis = True
            logger.info("✓ Redis cache initialized successfully.")
        except Exception as e:
            self.using_redis = False
            logger.warning(f"! Redis unavailable ({str(e)}). Active fallback: In-Memory Cache.")

    def _make_key(self, prefix: str, data: str) -> str:
        hashed = hashlib.sha256(data.encode('utf-8')).hexdigest()
        return f"verimind:{prefix}:{hashed}"

    def get(self, prefix: str, key_data: str) -> Optional[Any]:
        cache_key = self._make_key(prefix, key_data)
        if self.using_redis:
            try:
                val = self.redis_client.get(cache_key)
                if val:
                    return json.loads(val)
            except Exception:
                pass
        else:
            return self.memory_cache.get(cache_key)
        return None

    def set(self, prefix: str, key_data: str, value: Any, ttl_seconds: int = 3600):
        cache_key = self._make_key(prefix, key_data)
        serialized = json.dumps(value)
        if self.using_redis:
            try:
                self.redis_client.setex(cache_key, ttl_seconds, serialized)
            except Exception as e:
                logger.warning(f"Redis write error: {str(e)}")
        else:
            self.memory_cache[cache_key] = value

cache_manager = CacheManager()