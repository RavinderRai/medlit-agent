from datetime import date

from pydantic import BaseModel, Field, model_validator


class SearchFilters(BaseModel):
    """Filters for PubMed search."""

    min_date: date | None = Field(default=None, description="Minimum publication date")
    max_date: date | None = Field(default=None, description="Maximum publication date")
    article_types: list[str] = Field(
        default_factory=list,
        description="Filter by article types (e.g., 'Meta-Analysis', 'Clinical Trial')",
    )
    species: str = Field(default="humans", description="Species filter")
    language: str = Field(default="english", description="Language filter")

    @model_validator(mode="after")
    def validate_dates(self) -> "SearchFilters":
        """Ensure min_date is before max_date."""
        if self.min_date and self.max_date and self.min_date > self.max_date:
            raise ValueError("min_date must be before max_date")
        return self

    def to_pubmed_params(self) -> dict[str, str]:
        """Convert to PubMed API parameters."""
        params: dict[str, str] = {}

        if self.min_date:
            params["mindate"] = self.min_date.strftime("%Y/%m/%d")
        if self.max_date:
            params["maxdate"] = self.max_date.strftime("%Y/%m/%d")
        if self.min_date or self.max_date:
            params["datetype"] = "pdat"

        return params

    def to_query_filters(self) -> list[str]:
        """Convert to PubMed query filter strings."""
        filters = []

        if self.species:
            filters.append(f"{self.species}[filter]")
        if self.language:
            filters.append(f"{self.language}[la]")
        for article_type in self.article_types:
            filters.append(f"{article_type}[pt]")

        return filters


class SearchQuery(BaseModel):
    """Structured PubMed search query."""

    original_question: str = Field(..., description="Original user question")
    search_terms: list[str] = Field(
        default_factory=list,
        description="Extracted search terms",
    )
    mesh_terms: list[str] = Field(
        default_factory=list,
        description="MeSH terms for the query",
    )
    pubmed_query: str = Field(default="", description="Formatted PubMed query string")
    filters: SearchFilters = Field(
        default_factory=SearchFilters,
        description="Search filters",
    )
    max_results: int = Field(default=8, ge=1, le=50, description="Maximum results to return")

    def build_query(self) -> str:
        """Build the full PubMed query string with filters."""
        parts = []

        if self.pubmed_query:
            parts.append(f"({self.pubmed_query})")
        elif self.mesh_terms:
            mesh_part = " OR ".join(f"{term}[MeSH]" for term in self.mesh_terms)
            parts.append(f"({mesh_part})")
        elif self.search_terms:
            terms_part = " AND ".join(self.search_terms)
            parts.append(f"({terms_part})")

        filter_strings = self.filters.to_query_filters()
        parts.extend(filter_strings)

        return " AND ".join(parts)
