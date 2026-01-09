"""Caching utilities for MedLit agent."""

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from medlit.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Optional Redis import
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore


class Cache(ABC):
    """Abstract base class for caching."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached values."""
        pass

    @staticmethod
    def make_key(prefix: str, *args: Any) -> str:
        """Generate a cache key from arguments."""
        key_data = json.dumps(args, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"


class InMemoryCache(Cache):
    """Simple in-memory cache implementation."""

    def __init__(self, default_ttl: int = 3600):
        """Initialize in-memory cache.

        Args:
            default_ttl: Default TTL in seconds
        """
        self.default_ttl = default_ttl
        self._cache: dict[str, tuple[Any, datetime]] = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]

        if datetime.utcnow() > expiry:
            del self._cache[key]
            return None

        logger.debug("Cache hit", key=key)
        return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        self._cache[key] = (value, expiry)
        logger.debug("Cache set", key=key, ttl=ttl)

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        self._cache.pop(key, None)
        logger.debug("Cache delete", key=key)

    async def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        logger.debug("Cache cleared")

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        now = datetime.utcnow()
        expired = [k for k, (_, exp) in self._cache.items() if now > exp]
        for key in expired:
            del self._cache[key]
        return len(expired)


class RedisCache(Cache):
    """Redis-based cache implementation."""

    def __init__(
        self,
        redis_url: str,
        default_ttl: int = 3600,
        prefix: str = "medlit",
    ):
        """Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            prefix: Key prefix for namespacing
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed")

        self.default_ttl = default_ttl
        self.prefix = prefix
        self._client: Optional[redis.Redis] = None
        self._redis_url = redis_url

    async def _get_client(self) -> "redis.Redis":
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.from_url(self._redis_url)
        return self._client

    def _prefixed_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        client = await self._get_client()
        prefixed_key = self._prefixed_key(key)

        value = await client.get(prefixed_key)
        if value is None:
            return None

        logger.debug("Cache hit", key=key)
        return json.loads(value)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value in cache."""
        client = await self._get_client()
        prefixed_key = self._prefixed_key(key)
        ttl = ttl or self.default_ttl

        serialized = json.dumps(value, default=str)
        await client.setex(prefixed_key, ttl, serialized)
        logger.debug("Cache set", key=key, ttl=ttl)

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        client = await self._get_client()
        prefixed_key = self._prefixed_key(key)
        await client.delete(prefixed_key)
        logger.debug("Cache delete", key=key)

    async def clear(self) -> None:
        """Clear all cached values with our prefix."""
        client = await self._get_client()
        pattern = f"{self.prefix}:*"

        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break

        logger.debug("Cache cleared", prefix=self.prefix)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


# Global cache instance
_cache: Optional[Cache] = None


def get_cache() -> Cache:
    """Get or create the cache instance."""
    global _cache

    if _cache is not None:
        return _cache

    settings = get_settings()

    if settings.has_redis and REDIS_AVAILABLE:
        try:
            _cache = RedisCache(settings.redis_url)
            logger.info("Using Redis cache")
        except Exception as e:
            logger.warning("Failed to initialize Redis cache", error=str(e))
            _cache = InMemoryCache()
            logger.info("Falling back to in-memory cache")
    else:
        _cache = InMemoryCache()
        logger.info("Using in-memory cache")

    return _cache
