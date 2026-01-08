"""PubMed search functionality."""

from typing import TYPE_CHECKING

import structlog

from medlit.models import SearchQuery
from medlit.pubmed.parser import parse_search_results

if TYPE_CHECKING:
    from medlit.pubmed.client import PubMedClient

logger = structlog.get_logger(__name__)


async def search_pubmed(
    client: "PubMedClient",
    query: SearchQuery,
) -> list[str]:
    """Search PubMed for articles matching the query.

    Args:
        client: PubMed client instance
        query: Structured search query

    Returns:
        List of PMIDs matching the query
    """
    # Build the full query with filters
    full_query = query.build_query()

    if not full_query:
        logger.warning("Empty search query")
        return []

    # Prepare search parameters
    params = {
        "db": "pubmed",
        "term": full_query,
        "retmax": str(query.max_results),
        "sort": "relevance",
        "usehistory": "n",
    }

    # Add date filters if specified
    date_params = query.filters.to_pubmed_params()
    params.update(date_params)

    logger.info(
        "Searching PubMed",
        query=full_query,
        max_results=query.max_results,
    )

    # Execute search
    response_xml = await client._request("esearch.fcgi", params)

    # Parse results
    pmids = parse_search_results(response_xml)

    logger.info(
        "Search completed",
        query=full_query,
        results_count=len(pmids),
    )

    return pmids
