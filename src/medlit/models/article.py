from datetime import date

from pydantic import BaseModel, Field


class Author(BaseModel):
    """Author information from PubMed."""

    last_name: str = Field(..., description="Author's last name")
    first_name: str = Field(default="", description="Author's first name")
    initials: str = Field(default="", description="Author's initials")
    affiliation: str | None = Field(default=None, description="Author's affiliation")

    @property
    def full_name(self) -> str:
        """Get full name in 'Last, First' format."""
        if self.first_name:
            return f"{self.last_name}, {self.first_name}"
        return self.last_name

    @property
    def citation_name(self) -> str:
        """Get name in citation format 'Last FM'."""
        if self.initials:
            return f"{self.last_name} {self.initials}"
        return self.last_name


class Article(BaseModel):
    """PubMed article metadata and content."""

    pmid: str = Field(..., description="PubMed ID")
    title: str = Field(..., description="Article title")
    abstract: str = Field(default="", description="Article abstract")
    authors: list[Author] = Field(default_factory=list, description="List of authors")
    journal: str = Field(default="", description="Journal name")
    journal_abbrev: str = Field(default="", description="Journal abbreviation")
    publication_date: date | None = Field(default=None, description="Publication date")
    year: int | None = Field(default=None, description="Publication year")
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    article_type: str | None = Field(default=None, description="Type of article")
    mesh_terms: list[str] = Field(default_factory=list, description="MeSH terms")
    keywords: list[str] = Field(default_factory=list, description="Keywords")

    @property
    def pubmed_url(self) -> str:
        """Get PubMed URL for this article."""
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}"

    @property
    def first_author(self) -> str:
        """Get first author's citation name."""
        if self.authors:
            return self.authors[0].citation_name
        return "Unknown"

    @property
    def citation(self) -> str:
        """Get formatted citation string."""
        author_str = self.first_author
        if len(self.authors) > 1:
            author_str += " et al."

        parts = [author_str]
        if self.year:
            parts.append(f"({self.year})")
        parts.append(self.title)
        if self.journal_abbrev:
            parts.append(self.journal_abbrev)
        elif self.journal:
            parts.append(self.journal)

        return ". ".join(parts)

    def to_context_string(self) -> str:
        """Format article for LLM context."""
        lines = [
            f"**Title**: {self.title}",
            f"**Authors**: {self.first_author}" + (" et al." if len(self.authors) > 1 else ""),
            f"**Journal**: {self.journal or 'N/A'}",
            f"**Year**: {self.year or 'N/A'}",
            f"**PMID**: {self.pmid}",
        ]
        if self.article_type:
            lines.append(f"**Type**: {self.article_type}")
        lines.append(f"\n**Abstract**:\n{self.abstract or 'No abstract available.'}")

        return "\n".join(lines)
