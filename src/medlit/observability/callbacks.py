"""Custom callbacks for Google ADK and LangSmith integration."""

from datetime import datetime
from typing import Any

import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)


class MedLitCallbackHandler:
    """Callback handler for tracking agent events."""

    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def on_agent_start(
        self,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Called when agent starts processing a query."""
        self.start_time = datetime.utcnow()
        self._log_event("agent_start", {"query": query, **(metadata or {})})
        logger.info("Agent started", query=query[:100])

    def on_agent_end(
        self,
        result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Called when agent finishes processing."""
        self.end_time = datetime.utcnow()
        duration = None
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()

        self._log_event(
            "agent_end",
            {"duration_seconds": duration, **(metadata or {})},
        )
        logger.info("Agent finished", duration=duration)

    def on_tool_start(
        self,
        tool_name: str,
        tool_input: Any,
    ) -> None:
        """Called when a tool is invoked."""
        self._log_event(
            "tool_start",
            {"tool": tool_name, "input": str(tool_input)[:200]},
        )
        logger.debug("Tool started", tool=tool_name)

    def on_tool_end(
        self,
        tool_name: str,
        tool_output: Any,
    ) -> None:
        """Called when a tool completes."""
        self._log_event(
            "tool_end",
            {"tool": tool_name, "output": str(tool_output)[:200]},
        )
        logger.debug("Tool completed", tool=tool_name)

    def on_tool_error(
        self,
        tool_name: str,
        error: Exception,
    ) -> None:
        """Called when a tool errors."""
        self._log_event(
            "tool_error",
            {"tool": tool_name, "error": str(error)},
        )
        logger.error("Tool error", tool=tool_name, error=str(error))

    def on_llm_start(
        self,
        model: str,
        prompt_tokens: int | None = None,
    ) -> None:
        """Called when LLM inference starts."""
        self._log_event(
            "llm_start",
            {"model": model, "prompt_tokens": prompt_tokens},
        )
        logger.debug("LLM inference started", model=model)

    def on_llm_end(
        self,
        model: str,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Called when LLM inference completes."""
        self._log_event(
            "llm_end",
            {
                "model": model,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )
        logger.debug(
            "LLM inference completed",
            model=model,
            tokens=total_tokens,
        )

    def on_error(
        self,
        error: Exception,
        context: str | None = None,
    ) -> None:
        """Called on any error."""
        self._log_event(
            "error",
            {"error": str(error), "context": context},
        )
        logger.error("Error occurred", error=str(error), context=context)

    def _log_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Log an event internally."""
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }
        self.events.append(event)

    def get_events(self) -> list[dict[str, Any]]:
        """Get all logged events."""
        return self.events.copy()

    def clear_events(self) -> None:
        """Clear all logged events."""
        self.events.clear()
        self.start_time = None
        self.end_time = None
