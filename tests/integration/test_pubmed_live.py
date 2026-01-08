"""Live integration tests for PubMed API.

These tests make real API calls and are skipped in CI.
Run with: pytest tests/integration/ -m "not ci_skip"
"""

import pytest
from datetime import date

from medlit.models import SearchQuery, SearchFilters
from medlit.pubmed import PubMedClient


pytestmark = pytest.mark.skipif(
    True,  # Set to False to run live tests
    reason="Live API tests disabled by default",
)


class TestPubMedClientLive:
    """Live tests for PubMed client."""

    @pytest.fixture
    async def client(self):
        """Create PubMed client."""
        async with PubMedClient() as client:
            yield client

    @pytest.mark.asyncio
    async def test_search_returns_results(self, client):
        """Test that search returns PMIDs."""
        query = SearchQuery(
            original_question="aspirin cardiovascular",
            pubmed_query="aspirin[MeSH] AND cardiovascular[MeSH]",
            max_results=5,
        )

        pmids = await client.search(query)

        assert len(pmids) > 0
        assert all(pmid.isdigit() for pmid in pmids)

    @pytest.mark.asyncio
    async def test_fetch_returns_articles(self, client):
        """Test that fetch returns article details."""
        # Use known stable PMIDs
        pmids = ["33186500"]  # A real PMID

        articles = await client.fetch(pmids)

        assert len(articles) == 1
        article = articles[0]
        assert article.pmid == "33186500"
        assert article.title
        assert article.abstract

    @pytest.mark.asyncio
    async def test_search_and_fetch(self, client):
        """Test combined search and fetch."""
        today = date.today()
        min_date = date(today.year - 2, 1, 1)

        query = SearchQuery(
            original_question="COVID-19 vaccine efficacy",
            pubmed_query="COVID-19[MeSH] AND vaccine[MeSH] AND efficacy[tiab]",
            filters=SearchFilters(min_date=min_date, max_date=today),
            max_results=3,
        )

        articles = await client.search_and_fetch(query)

        assert len(articles) > 0
        for article in articles:
            assert article.pmid
            assert article.title
