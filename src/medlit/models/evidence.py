from enum import Enum

from pydantic import BaseModel, Field


class EvidenceQuality(str, Enum):
    """Quality level of evidence based on study design."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"

    @classmethod
    def from_article_type(cls, article_type: str | None) -> "EvidenceQuality":
        """Infer evidence quality from article type."""
        if not article_type:
            return cls.UNKNOWN

        article_type_lower = article_type.lower()

        if any(t in article_type_lower for t in ["meta-analysis", "systematic review"]):
            return cls.HIGH
        elif any(t in article_type_lower for t in ["randomized", "rct", "clinical trial"]):
            return cls.MODERATE
        elif any(t in article_type_lower for t in ["cohort", "case-control", "case report", "case series", "review"]):
            return cls.LOW
        else:
            return cls.UNKNOWN


class Citation(BaseModel):
    """Citation reference for evidence."""

    pmid: str = Field(..., description="PubMed ID")
    title: str = Field(..., description="Article title")
    authors: str = Field(default="", description="Author string")
    year: int | None = Field(default=None, description="Publication year")
    journal: str = Field(default="", description="Journal name")

    @property
    def url(self) -> str:
        """Get PubMed URL."""
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}"

    def to_markdown(self) -> str:
        """Format as markdown link."""
        label = f"{self.authors}"
        if self.year:
            label += f" ({self.year})"
        return f"[{label}]({self.url})"

    def to_inline_citation(self) -> str:
        """Format as inline citation."""
        if self.year:
            return f"[PMID: {self.pmid}, {self.year}]"
        return f"[PMID: {self.pmid}]"


class Evidence(BaseModel):
    """Synthesized evidence from literature."""

    summary: str = Field(..., description="Summary of the evidence")
    quality: EvidenceQuality = Field(
        default=EvidenceQuality.UNKNOWN,
        description="Overall quality of evidence",
    )
    supporting_citations: list[Citation] = Field(
        default_factory=list,
        description="Citations supporting the summary",
    )
    conflicting_citations: list[Citation] = Field(
        default_factory=list,
        description="Citations with conflicting findings",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations of the evidence",
    )
    consensus: str | None = Field(
        default=None,
        description="Description of scientific consensus if present",
    )

    @property
    def total_citations(self) -> int:
        """Get total number of citations."""
        return len(self.supporting_citations) + len(self.conflicting_citations)

    @property
    def has_conflicts(self) -> bool:
        """Check if there are conflicting findings."""
        return len(self.conflicting_citations) > 0
