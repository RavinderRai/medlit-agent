"""Tests for data models."""

import pytest
from datetime import date

from medlit.models import Article, Author, SearchQuery, SearchFilters, Evidence, Citation, EvidenceQuality


class TestAuthor:
    """Tests for Author model."""

    def test_full_name(self):
        """Test full name formatting."""
        author = Author(last_name="Smith", first_name="John", initials="J")
        assert author.full_name == "Smith, John"

    def test_full_name_no_first(self):
        """Test full name with no first name."""
        author = Author(last_name="Smith", initials="J")
        assert author.full_name == "Smith"

    def test_citation_name(self):
        """Test citation name formatting."""
        author = Author(last_name="Smith", first_name="John", initials="JA")
        assert author.citation_name == "Smith JA"


class TestArticle:
    """Tests for Article model."""

    def test_pubmed_url(self):
        """Test PubMed URL generation."""
        article = Article(pmid="12345678", title="Test Article")
        assert article.pubmed_url == "https://pubmed.ncbi.nlm.nih.gov/12345678"

    def test_first_author(self):
        """Test first author extraction."""
        article = Article(
            pmid="12345678",
            title="Test",
            authors=[
                Author(last_name="Smith", initials="J"),
                Author(last_name="Jones", initials="M"),
            ],
        )
        assert article.first_author == "Smith J"

    def test_first_author_none(self):
        """Test first author when no authors."""
        article = Article(pmid="12345678", title="Test")
        assert article.first_author == "Unknown"

    def test_citation(self):
        """Test citation string generation."""
        article = Article(
            pmid="12345678",
            title="Effects of Aspirin",
            authors=[Author(last_name="Smith", initials="J")],
            year=2023,
            journal_abbrev="JAMA",
        )
        citation = article.citation
        assert "Smith J" in citation
        assert "2023" in citation
        assert "JAMA" in citation


class TestSearchFilters:
    """Tests for SearchFilters model."""

    def test_date_validation(self):
        """Test date validation."""
        with pytest.raises(ValueError):
            SearchFilters(
                min_date=date(2024, 1, 1),
                max_date=date(2020, 1, 1),
            )

    def test_to_query_filters(self):
        """Test filter string generation."""
        filters = SearchFilters(species="humans", language="english")
        query_filters = filters.to_query_filters()
        assert "humans[filter]" in query_filters
        assert "english[la]" in query_filters


class TestSearchQuery:
    """Tests for SearchQuery model."""

    def test_build_query_with_pubmed_query(self):
        """Test query building with pre-formatted query."""
        query = SearchQuery(
            original_question="test",
            pubmed_query="aspirin[MeSH]",
        )
        result = query.build_query()
        assert "aspirin[MeSH]" in result

    def test_build_query_with_mesh_terms(self):
        """Test query building with MeSH terms."""
        query = SearchQuery(
            original_question="test",
            mesh_terms=["Aspirin", "Cardiovascular Diseases"],
        )
        result = query.build_query()
        assert "Aspirin[MeSH]" in result
        assert "Cardiovascular Diseases[MeSH]" in result

    def test_build_query_with_filters(self):
        """Test query building includes filters."""
        query = SearchQuery(
            original_question="test",
            pubmed_query="aspirin",
            filters=SearchFilters(species="humans"),
        )
        result = query.build_query()
        assert "humans[filter]" in result


class TestEvidenceQuality:
    """Tests for EvidenceQuality enum."""

    def test_from_meta_analysis(self):
        """Test quality inference from meta-analysis."""
        quality = EvidenceQuality.from_article_type("Meta-Analysis")
        assert quality == EvidenceQuality.HIGH

    def test_from_rct(self):
        """Test quality inference from RCT."""
        quality = EvidenceQuality.from_article_type("Randomized Controlled Trial")
        assert quality == EvidenceQuality.MODERATE

    def test_from_unknown(self):
        """Test quality inference from unknown type."""
        quality = EvidenceQuality.from_article_type("Letter")
        assert quality == EvidenceQuality.UNKNOWN


class TestCitation:
    """Tests for Citation model."""

    def test_url(self):
        """Test URL generation."""
        citation = Citation(pmid="12345678", title="Test")
        assert citation.url == "https://pubmed.ncbi.nlm.nih.gov/12345678"

    def test_to_markdown(self):
        """Test markdown formatting."""
        citation = Citation(
            pmid="12345678",
            title="Test",
            authors="Smith J",
            year=2023,
        )
        md = citation.to_markdown()
        assert "[Smith J (2023)]" in md
        assert "12345678" in md

    def test_inline_citation(self):
        """Test inline citation formatting."""
        citation = Citation(pmid="12345678", title="Test", year=2023)
        inline = citation.to_inline_citation()
        assert "PMID: 12345678" in inline
        assert "2023" in inline
