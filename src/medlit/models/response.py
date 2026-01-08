from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from medlit.models.evidence import Citation, Evidence


class ResponseStatus(str, Enum):
    """Status of the agent response."""

    SUCCESS = "success"
    NO_RESULTS = "no_results"
    ERROR = "error"
    PARTIAL = "partial"


class AgentResponse(BaseModel):
    """Complete response from the MedLit agent."""

    question: str = Field(..., description="Original user question")
    status: ResponseStatus = Field(default=ResponseStatus.SUCCESS, description="Response status")
    answer: str = Field(default="", description="Synthesized answer to the question")
    evidence: Evidence | None = Field(default=None, description="Supporting evidence")
    citations: list[Citation] = Field(
        default_factory=list,
        description="All citations used in the response",
    )
    pubmed_query: str = Field(default="", description="PubMed query used")
    articles_found: int = Field(default=0, description="Number of articles found")
    articles_analyzed: int = Field(default=0, description="Number of articles analyzed")
    error_message: str | None = Field(default=None, description="Error message if status is error")
    disclaimer: str = Field(
        default="",
        description="Medical disclaimer",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp",
    )
    trace_id: str | None = Field(default=None, description="LangSmith trace ID")

    def to_markdown(self) -> str:
        """Format response as markdown."""
        parts = []

        # Answer section
        if self.answer:
            parts.append(f"## Answer\n\n{self.answer}")
        elif self.status == ResponseStatus.NO_RESULTS:
            parts.append("## Answer\n\nNo relevant articles found for this query.")
        elif self.status == ResponseStatus.ERROR:
            parts.append(f"## Error\n\n{self.error_message or 'An error occurred.'}")

        # Evidence quality
        if self.evidence:
            parts.append(f"\n**Evidence Quality**: {self.evidence.quality.value.replace('_', ' ').title()}")

            if self.evidence.limitations:
                parts.append("\n**Limitations**:")
                for limitation in self.evidence.limitations:
                    parts.append(f"- {limitation}")

        # Citations
        if self.citations:
            parts.append("\n## Sources")
            for i, citation in enumerate(self.citations, 1):
                parts.append(f"{i}. {citation.to_markdown()}")

        # Search info
        parts.append(f"\n---\n*Query: `{self.pubmed_query}`*")
        parts.append(f"*Articles found: {self.articles_found}, analyzed: {self.articles_analyzed}*")

        # Disclaimer
        if self.disclaimer:
            parts.append(f"\n{self.disclaimer}")

        return "\n".join(parts)

    def to_text(self) -> str:
        """Format response as plain text."""
        parts = []

        if self.answer:
            parts.append(f"ANSWER:\n{self.answer}")
        elif self.status == ResponseStatus.NO_RESULTS:
            parts.append("ANSWER:\nNo relevant articles found for this query.")
        elif self.status == ResponseStatus.ERROR:
            parts.append(f"ERROR:\n{self.error_message or 'An error occurred.'}")

        if self.citations:
            parts.append("\nSOURCES:")
            for citation in self.citations:
                parts.append(f"- {citation.url}")

        if self.disclaimer:
            parts.append(f"\n{self.disclaimer}")

        return "\n".join(parts)
