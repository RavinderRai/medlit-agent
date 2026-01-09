"""PubMed E-utilities API client."""

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from medlit.config.constants import (
    PUBMED_BASE_URL,
    PUBMED_RATE_LIMIT,
    PUBMED_RATE_LIMIT_WITH_KEY,
)
from medlit.config.settings import get_settings
from medlit.models import Article, SearchQuery
from medlit.pubmed.fetch import fetch_articles
from medlit.pubmed.search import search_pubmed
from medlit.utils.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)


class PubMedClient:
    """Client for interacting with PubMed E-utilities API."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize PubMed client.

        Args:
            api_key: NCBI API key (optional, increases rate limit)
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        self.api_key = api_key or settings.ncbi_api_key
        self.base_url = PUBMED_BASE_URL
        self.timeout = timeout

        # Set rate limit based on API key availability
        rate_limit = PUBMED_RATE_LIMIT_WITH_KEY if self.api_key else PUBMED_RATE_LIMIT
        self.rate_limiter = RateLimiter(rate_limit)

        self._client: httpx.AsyncClient | None = None

        logger.info(
            "PubMed client initialized",
            has_api_key=bool(self.api_key),
            rate_limit=rate_limit,
        )

    async def __aenter__(self) -> "PubMedClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client, creating if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _get_base_params(self) -> dict[str, str]:
        """Get base parameters for all requests."""
        params = {"retmode": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _request(
        self,
        endpoint: str,
        params: dict[str, str],
    ) -> str:
        """Make a rate-limited request to PubMed API.

        Args:
            endpoint: API endpoint (e.g., 'esearch.fcgi')
            params: Query parameters

        Returns:
            Response text
        """
        await self.rate_limiter.acquire()

        url = f"{self.base_url}/{endpoint}"
        all_params = {**self._get_base_params(), **params}

        logger.debug("PubMed request", endpoint=endpoint, params=params)

        response = await self.client.get(url, params=all_params)
        response.raise_for_status()

        return response.text

    async def search(self, query: SearchQuery) -> list[str]:
        """Search PubMed and return PMIDs.

        Args:
            query: Structured search query

        Returns:
            List of PMIDs matching the query
        """
        return await search_pubmed(self, query)

    async def fetch(self, pmids: list[str]) -> list[Article]:
        """Fetch article details for given PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of Article objects
        """
        return await fetch_articles(self, pmids)

    async def search_and_fetch(self, query: SearchQuery) -> list[Article]:
        """Search PubMed and fetch full article details.

        Convenience method that combines search and fetch.

        Args:
            query: Structured search query

        Returns:
            List of Article objects
        """
        pmids = await self.search(query)
        if not pmids:
            logger.info("No results found", query=query.pubmed_query)
            return []

        return await self.fetch(pmids)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
