"""Async rate limiter for API calls."""

import asyncio
import time

import structlog

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter for async operations."""

    def __init__(
        self,
        rate: float,
        burst: int | None = None,
    ):
        """Initialize rate limiter.

        Args:
            rate: Maximum requests per second
            burst: Maximum burst size (defaults to rate)
        """
        self.rate = rate
        self.burst = burst or int(rate)
        self.tokens = float(self.burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        async with self._lock:
            waited = 0.0

            while True:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return waited

                # Calculate wait time
                needed = tokens - self.tokens
                wait_time = needed / self.rate

                logger.debug(
                    "Rate limit: waiting",
                    wait_seconds=wait_time,
                    tokens_needed=needed,
                )

                await asyncio.sleep(wait_time)
                waited += wait_time

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now

        # Add tokens based on elapsed time
        self.tokens = min(
            self.burst,
            self.tokens + elapsed * self.rate,
        )

    async def __aenter__(self) -> "RateLimiter":
        """Async context manager entry - acquires one token."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (without acquiring)."""
        self._refill()
        return self.tokens


class ConcurrencyLimiter:
    """Semaphore-based concurrency limiter."""

    def __init__(self, max_concurrent: int):
        """Initialize concurrency limiter.

        Args:
            max_concurrent: Maximum concurrent operations
        """
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._current = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a slot."""
        await self._semaphore.acquire()
        async with self._lock:
            self._current += 1
            logger.debug(
                "Concurrency slot acquired",
                current=self._current,
                max=self.max_concurrent,
            )

    def release(self) -> None:
        """Release a slot."""
        self._semaphore.release()
        # Note: _current update is approximate here
        logger.debug("Concurrency slot released")

    async def __aenter__(self) -> "ConcurrencyLimiter":
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.release()

    @property
    def current_concurrent(self) -> int:
        """Get current number of concurrent operations."""
        return self.max_concurrent - self._semaphore._value
