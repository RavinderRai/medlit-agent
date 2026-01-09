"""LangSmith observability integration with OpenTelemetry for Google ADK."""

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
    from langsmith.integrations.otel import configure as configure_otel
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    LangSmithClient = None
    traceable = None
    configure_otel = None

# Optional OpenTelemetry imports (for custom spans)
try:
    from opentelemetry import trace
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

# Optional OpenInference ADK instrumentation
try:
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor
    OPENINFERENCE_AVAILABLE = True
except ImportError:
    OPENINFERENCE_AVAILABLE = False
    GoogleADKInstrumentor = None

_langsmith_client: LangSmithClient | None = None
_tracing_configured: bool = False


def init_langsmith() -> bool:
    """Initialize LangSmith client and OpenTelemetry tracing for Google ADK.

    This uses LangSmith's built-in OpenTelemetry integration which automatically
    instruments Google ADK spans.

    Returns:
        True if LangSmith was initialized successfully
    """
    global _langsmith_client, _tracing_configured

    settings = get_settings()

    if not LANGSMITH_AVAILABLE:
        logger.warning("LangSmith not installed, tracing disabled")
        return False

    if not settings.langsmith_enabled:
        logger.info("LangSmith tracing not enabled")
        return False

    if not settings.langsmith_api_key:
        logger.warning("LangSmith API key not set, tracing disabled")
        return False

    try:
        # Set environment variables for LangSmith
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

        # Configure LangSmith OpenTelemetry integration
        # This sets up the OTel pipeline to send traces to LangSmith
        if configure_otel is not None:
            configure_otel(project_name=settings.langsmith_project)
            logger.info(
                "LangSmith OpenTelemetry pipeline configured",
                project=settings.langsmith_project,
            )

        # Instrument Google ADK to capture spans
        # This is required to actually capture ADK agent/tool/LLM data
        if OPENINFERENCE_AVAILABLE and GoogleADKInstrumentor is not None:
            GoogleADKInstrumentor().instrument()
            _tracing_configured = True
            logger.info("Google ADK instrumented for tracing")
        else:
            logger.warning(
                "openinference-instrumentation-google-adk not installed, "
                "ADK spans will not be captured. Install with: "
                "uv add openinference-instrumentation-google-adk"
            )

        # Also create client for direct API access if needed
        _langsmith_client = LangSmithClient(api_key=settings.langsmith_api_key)
        logger.info(
            "LangSmith client initialized",
            project=settings.langsmith_project,
        )

        return True
    except Exception as e:
        logger.error("Failed to initialize LangSmith", error=str(e))
        return False


def get_langsmith_client() -> LangSmithClient | None:
    """Get the LangSmith client instance."""
    return _langsmith_client


def is_tracing_configured() -> bool:
    """Check if tracing has been configured."""
    return _tracing_configured


def get_tracer(name: str = __name__) -> trace.Tracer | None:
    """Get an OpenTelemetry tracer instance for custom spans.

    Args:
        name: Name for the tracer (defaults to module name)

    Returns:
        OpenTelemetry tracer or None if not available
    """
    if OTEL_AVAILABLE and trace is not None and _tracing_configured:
        return trace.get_tracer(name)
    return None


F = TypeVar("F", bound=Callable[..., Any])


def trace_function(
    name: str | None = None,
    run_type: str = "chain",
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator to trace a function with LangSmith.

    Note: For Google ADK, most tracing is automatic via configure().
    This decorator is for custom spans outside of ADK.

    Args:
        name: Name for the trace (defaults to function name)
        run_type: Type of run (chain, tool, llm, retriever)
        metadata: Additional metadata to attach to the trace

    Returns:
        Decorated function (works with both sync and async)
    """
    import asyncio

    def decorator(func: F) -> F:
        trace_name = name or func.__name__

        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                settings = get_settings()

                # Use OpenTelemetry tracing if configured
                tracer = get_tracer()
                if tracer is not None and settings.langsmith_enabled:
                    with tracer.start_as_current_span(
                        trace_name,
                        attributes={
                            "run_type": run_type,
                            **(metadata or {}),
                        },
                    ):
                        return await func(*args, **kwargs)

                # Fall back to LangSmith traceable
                if LANGSMITH_AVAILABLE and traceable is not None and settings.langsmith_enabled:
                    traced_func = traceable(
                        name=trace_name,
                        run_type=run_type,
                        metadata=metadata or {},
                    )(func)
                    return await traced_func(*args, **kwargs)

                return await func(*args, **kwargs)
            return async_wrapper  # type: ignore
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                settings = get_settings()

                # Use OpenTelemetry tracing if configured
                tracer = get_tracer()
                if tracer is not None and settings.langsmith_enabled:
                    with tracer.start_as_current_span(
                        trace_name,
                        attributes={
                            "run_type": run_type,
                            **(metadata or {}),
                        },
                    ):
                        return func(*args, **kwargs)

                # Fall back to LangSmith traceable
                if LANGSMITH_AVAILABLE and traceable is not None and settings.langsmith_enabled:
                    traced_func = traceable(
                        name=trace_name,
                        run_type=run_type,
                        metadata=metadata or {},
                    )(func)
                    return traced_func(*args, **kwargs)

                return func(*args, **kwargs)
            return sync_wrapper  # type: ignore

    return decorator


class TracingContext:
    """Context manager for creating a custom trace span.

    Note: For Google ADK, most tracing is automatic via configure().
    Use this for custom spans outside of ADK.
    """

    def __init__(
        self,
        name: str,
        run_type: str = "chain",
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.run_type = run_type
        self.metadata = metadata or {}
        self._span: Any | None = None
        self._token: Any | None = None

    def __enter__(self) -> TracingContext:
        """Start the trace span."""
        settings = get_settings()
        if not settings.langsmith_enabled:
            return self

        tracer = get_tracer()
        if tracer is not None:
            try:
                self._span = tracer.start_span(
                    self.name,
                    attributes={
                        "run_type": self.run_type,
                        **self.metadata,
                    },
                )
                # Make span current
                from opentelemetry.trace import use_span
                self._token = use_span(self._span, end_on_exit=False).__enter__()
                logger.debug("Started trace span", name=self.name)
            except Exception as e:
                logger.warning("Failed to start trace span", error=str(e))

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End the trace span."""
        if self._span is not None:
            try:
                if exc_type is not None and trace is not None:
                    self._span.set_status(
                        trace.Status(trace.StatusCode.ERROR, str(exc_val))
                    )
                    self._span.record_exception(exc_val)
                self._span.end()
                logger.debug("Ended trace span", name=self.name)
            except Exception as e:
                logger.warning("Failed to end trace span", error=str(e))

    @property
    def span(self) -> Any | None:
        """Get the current span."""
        return self._span
