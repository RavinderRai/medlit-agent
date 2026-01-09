"""Evidence fetching tool for Google ADK agent."""

import structlog
from google.adk.tools import FunctionTool
from langsmith import traceable

from medlit.pubmed import PubMedClient

logger = structlog.get_logger(__name__)


@traceable(name="fetch_evidence", run_type="tool")
async def fetch_evidence(pmids: list[str]) -> dict:
    """Fetch full article details from PubMed for a list of PMIDs.

    Use this tool after searching PubMed to get the full abstracts and metadata
    for articles you want to analyze.

    Args:
        pmids: List of PubMed IDs (PMIDs) as strings (max 20)

    Returns:
        Dictionary with article details including title, abstract, authors, journal, year.
        Maximum 20 articles per request.
    """
    if not pmids:
        return {
            "success": False,
            "error": "No PMIDs provided",
            "articles": [],
        }

    # Limit to prevent excessive fetching
    pmids = pmids[:20]

    logger.info("Fetch evidence tool called", pmid_count=len(pmids))

    try:
        async with PubMedClient() as client:
            articles = await client.fetch(pmids)

            # Convert to serializable format
            article_data = []
            for article in articles:
                article_data.append({
                    "pmid": article.pmid,
                    "title": article.title,
                    "abstract": article.abstract,
                    "authors": article.first_author + (" et al." if len(article.authors) > 1 else ""),
                    "journal": article.journal,
                    "year": article.year,
                    "article_type": article.article_type,
                    "pubmed_url": article.pubmed_url,
                })

            return {
                "success": True,
                "articles": article_data,
                "count": len(article_data),
            }

    except Exception as e:
        logger.error("Evidence fetch failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "articles": [],
        }


# Create the tool for Google ADK
# Note: FunctionTool automatically extracts name from func.__name__ and description from func.__doc__
fetch_evidence_tool = FunctionTool(func=fetch_evidence)
