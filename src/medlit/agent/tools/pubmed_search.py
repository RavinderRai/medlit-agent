"""PubMed search tool for Google ADK agent."""

from datetime import date

import structlog
from google.adk.tools import FunctionTool

from medlit.config.constants import DEFAULT_DATE_RANGE_YEARS, DEFAULT_MAX_RESULTS
from medlit.models import SearchFilters, SearchQuery
from medlit.pubmed import PubMedClient

logger = structlog.get_logger(__name__)


async def search_pubmed(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    years_back: int = DEFAULT_DATE_RANGE_YEARS,
    article_types: list[str] | None = None,
) -> dict:
    """Search PubMed for medical research articles.

    Use this tool to find relevant medical literature for a question.
    You can use MeSH terms for more precise searches.

    Examples:
    - "aspirin[MeSH] AND primary prevention[MeSH]"
    - "metformin AND diabetes AND (efficacy OR effectiveness)"
    - "COVID-19[MeSH] AND vaccine[MeSH] AND adverse effects[sh]"

    Args:
        query: The PubMed search query string (can include MeSH terms)
        max_results: Maximum number of results to return (1-50)
        years_back: How many years back to search
        article_types: Optional filter for article types (e.g., ["Meta-Analysis", "Clinical Trial"])

    Returns:
        Dictionary with search results including PMIDs and count.
        Returns a list of PubMed IDs (PMIDs) that can be fetched for full details.
    """
    logger.info("PubMed search tool called", query=query, max_results=max_results)

    # Build date filters
    today = date.today()
    min_date = date(today.year - years_back, today.month, today.day)

    filters = SearchFilters(
        min_date=min_date,
        max_date=today,
        article_types=article_types or [],
    )

    search_query = SearchQuery(
        original_question=query,
        pubmed_query=query,
        filters=filters,
        max_results=min(max_results, 50),
    )

    try:
        async with PubMedClient() as client:
            pmids = await client.search(search_query)

            return {
                "success": True,
                "pmids": pmids,
                "count": len(pmids),
                "query_used": search_query.build_query(),
            }

    except Exception as e:
        logger.error("PubMed search failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "pmids": [],
            "count": 0,
        }


# Create the tool for Google ADK
# Note: FunctionTool automatically extracts name from func.__name__ and description from func.__doc__
search_pubmed_tool = FunctionTool(func=search_pubmed)
