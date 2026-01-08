"""PubMed article fetching functionality."""

from typing import TYPE_CHECKING

import structlog

from medlit.models import Article
from medlit.pubmed.parser import parse_articles

if TYPE_CHECKING:
    from medlit.pubmed.client import PubMedClient

logger = structlog.get_logger(__name__)


async def fetch_articles(
    client: "PubMedClient",
    pmids: list[str],
) -> list[Article]:
    """Fetch full article details from PubMed.

    Args:
        client: PubMed client instance
        pmids: List of PubMed IDs to fetch

    Returns:
        List of Article objects with full metadata
    """
    if not pmids:
        return []

    # Prepare fetch parameters
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
    }

    logger.info(
        "Fetching articles",
        pmid_count=len(pmids),
    )

    # Execute fetch
    response_xml = await client._request("efetch.fcgi", params)

    # Parse articles
    articles = parse_articles(response_xml)

    logger.info(
        "Articles fetched",
        requested=len(pmids),
        received=len(articles),
    )

    return articles
