"""LangSmith observability integration."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import structlog

from medlit.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Optional LangSmith import
try:
    from langsmith import Client as LangSmithClient
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    LangSmithClient = None
    traceable = None

_langsmith_client: "LangSmithClient" | None = None


def init_langsmith() -> bool:
    """Initialize LangSmith client if configured.

    Returns:
        True if LangSmith was initialized successfully
    """
    global _langsmith_client

    settings = get_settings()

    if not LANGSMITH_AVAILABLE:
        logger.warning("LangSmith not installed, tracing disabled")
        return False

    if not settings.langsmith_enabled:
        logger.info("LangSmith tracing not enabled")
        return False

    # Set environment variables for LangSmith
    # Note: LangSmith SDK reads from LANGCHAIN_* env vars
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    try:
        _langsmith_client = LangSmithClient(api_key=settings.langsmith_api_key)
        logger.info(
            "LangSmith initialized",
            project=settings.langsmith_project,
            api_key_set=bool(settings.langsmith_api_key),
        )
        return True
    except Exception as e:
        logger.error("Failed to initialize LangSmith", error=str(e))
        return False


def get_langsmith_client() -> "LangSmithClient" | None:
    """Get the LangSmith client instance."""
    return _langsmith_client


F = TypeVar("F", bound=Callable[..., Any])


def trace_function(
    name: str | None = None,
    run_type: str = "chain",
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator to trace a function with LangSmith.

    Args:
        name: Name for the trace (defaults to function name)
        run_type: Type of run (chain, tool, llm, retriever)
        metadata: Additional metadata to attach to the trace

    Returns:
        Decorated function (works with both sync and async)
    """
    import asyncio

    def decorator(func: F) -> F:
        if not LANGSMITH_AVAILABLE or traceable is None:
            return func

        # Apply LangSmith traceable decorator
        traced_func = traceable(
            name=name or func.__name__,
            run_type=run_type,
            metadata=metadata or {},
        )(func)

        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                settings = get_settings()
                if settings.langsmith_enabled:
                    return await traced_func(*args, **kwargs)
                return await func(*args, **kwargs)
            return async_wrapper  # type: ignore
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                settings = get_settings()
                if settings.langsmith_enabled:
                    return traced_func(*args, **kwargs)
                return func(*args, **kwargs)
            return sync_wrapper  # type: ignore

    return decorator


class TracingContext:
    """Context manager for creating a trace span."""

    def __init__(
        self,
        name: str,
        run_type: str = "chain",
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.run_type = run_type
        self.metadata = metadata or {}
        self._run_id: str | None = None

    def __enter__(self) -> "TracingContext":
        """Start the trace span."""
        settings = get_settings()
        if not settings.langsmith_enabled or _langsmith_client is None:
            return self

        try:
            # Note: In actual implementation, you'd use the proper
            # LangSmith context management APIs
            logger.debug("Starting trace span", name=self.name)
        except Exception as e:
            logger.warning("Failed to start trace span", error=str(e))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the trace span."""
        if exc_type is not None:
            logger.debug(
                "Trace span ended with error",
                name=self.name,
                error=str(exc_val),
            )
        else:
            logger.debug("Trace span ended", name=self.name)

    @property
    def run_id(self) -> str | None:
        """Get the run ID for this trace."""
        return self._run_id
