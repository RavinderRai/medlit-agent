"""Custom metrics tracking for MedLit agent."""

from dataclasses import dataclass, field
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""

    query: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None

    # Search metrics
    pubmed_query: str = ""
    articles_found: int = 0
    articles_analyzed: int = 0

    # LLM metrics
    llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # Latency metrics
    search_latency_ms: float | None = None
    fetch_latency_ms: float | None = None
    synthesis_latency_ms: float | None = None

    # Result metrics
    status: str = "pending"
    citations_count: int = 0
    evidence_quality: str | None = None

    @property
    def total_latency_ms(self) -> float | None:
        """Calculate total latency in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.total_prompt_tokens + self.total_completion_tokens

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/export."""
        return {
            "query": self.query[:100],
            "status": self.status,
            "total_latency_ms": self.total_latency_ms,
            "search_latency_ms": self.search_latency_ms,
            "fetch_latency_ms": self.fetch_latency_ms,
            "synthesis_latency_ms": self.synthesis_latency_ms,
            "articles_found": self.articles_found,
            "articles_analyzed": self.articles_analyzed,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "citations_count": self.citations_count,
            "evidence_quality": self.evidence_quality,
        }


class MetricsTracker:
    """Track and aggregate metrics across queries."""

    def __init__(self):
        self._current_metrics: QueryMetrics | None = None
        self._history: list[QueryMetrics] = []

    def start_query(self, query: str) -> QueryMetrics:
        """Start tracking a new query."""
        self._current_metrics = QueryMetrics(query=query)
        logger.debug("Started tracking query", query=query[:50])
        return self._current_metrics

    def end_query(self, status: str = "success") -> QueryMetrics | None:
        """End tracking for current query."""
        if self._current_metrics is None:
            return None

        self._current_metrics.end_time = datetime.utcnow()
        self._current_metrics.status = status
        self._history.append(self._current_metrics)

        logger.info(
            "Query completed",
            **self._current_metrics.to_dict(),
        )

        metrics = self._current_metrics
        self._current_metrics = None
        return metrics

    @property
    def current(self) -> QueryMetrics | None:
        """Get current query metrics."""
        return self._current_metrics

    def record_search(
        self,
        pubmed_query: str,
        articles_found: int,
        latency_ms: float,
    ) -> None:
        """Record search metrics."""
        if self._current_metrics:
            self._current_metrics.pubmed_query = pubmed_query
            self._current_metrics.articles_found = articles_found
            self._current_metrics.search_latency_ms = latency_ms

    def record_fetch(
        self,
        articles_analyzed: int,
        latency_ms: float,
    ) -> None:
        """Record fetch metrics."""
        if self._current_metrics:
            self._current_metrics.articles_analyzed = articles_analyzed
            self._current_metrics.fetch_latency_ms = latency_ms

    def record_synthesis(
        self,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record synthesis metrics."""
        if self._current_metrics:
            self._current_metrics.synthesis_latency_ms = latency_ms
            self._current_metrics.llm_calls += 1
            self._current_metrics.total_prompt_tokens += prompt_tokens
            self._current_metrics.total_completion_tokens += completion_tokens

    def record_result(
        self,
        citations_count: int,
        evidence_quality: str | None = None,
    ) -> None:
        """Record result metrics."""
        if self._current_metrics:
            self._current_metrics.citations_count = citations_count
            self._current_metrics.evidence_quality = evidence_quality

    def get_history(self) -> list[QueryMetrics]:
        """Get all historical metrics."""
        return self._history.copy()

    def get_summary(self) -> dict:
        """Get summary statistics across all queries."""
        if not self._history:
            return {"total_queries": 0}

        total_queries = len(self._history)
        successful = sum(1 for m in self._history if m.status == "success")

        latencies = [m.total_latency_ms for m in self._history if m.total_latency_ms]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        total_tokens = sum(m.total_tokens for m in self._history)
        total_articles = sum(m.articles_analyzed for m in self._history)

        return {
            "total_queries": total_queries,
            "successful_queries": successful,
            "success_rate": successful / total_queries if total_queries > 0 else 0,
            "avg_latency_ms": avg_latency,
            "total_tokens_used": total_tokens,
            "total_articles_analyzed": total_articles,
        }

    def clear_history(self) -> None:
        """Clear metrics history."""
        self._history.clear()
