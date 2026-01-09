"""Evidence synthesis tool for Google ADK agent."""

import json

import structlog
from google.adk.tools import FunctionTool
from langsmith import traceable

logger = structlog.get_logger(__name__)


@traceable(name="synthesize_evidence", run_type="tool")
async def synthesize_evidence(
    question: str,
    articles_json: str,
) -> dict:
    """Synthesize evidence from multiple research articles into a coherent summary.

    Use this tool after fetching article details to combine findings across
    multiple studies and assess the overall quality of evidence.

    Args:
        question: The medical question being answered
        articles_json: JSON string containing list of article objects from fetch_evidence

    Returns:
        Dictionary with synthesis results including summary, evidence quality, and citations.
    """
    # Parse the JSON string
    try:
        articles = json.loads(articles_json) if isinstance(articles_json, str) else articles_json
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": "Invalid JSON in articles_json",
            "synthesis": None,
        }

    if not articles:
        return {
            "success": False,
            "error": "No articles provided for synthesis",
            "synthesis": None,
        }

    logger.info(
        "Synthesize evidence tool called",
        question=question[:100],
        article_count=len(articles),
    )

    try:
        # Analyze study types
        study_types = {}
        for article in articles:
            article_type = article.get("article_type", "Unknown")
            study_types[article_type] = study_types.get(article_type, 0) + 1

        # Determine evidence quality based on study types
        quality = "unknown"
        if "Meta-Analysis" in study_types or "Systematic Review" in study_types:
            quality = "high"
        elif "Randomized Controlled Trial" in study_types or "Clinical Trial" in study_types:
            quality = "moderate"
        elif any(t in study_types for t in ["Cohort", "Case-Control", "Review", "Case Report"]):
            quality = "low"

        # Build synthesis summary (basic version - the main agent does detailed synthesis)
        summary_parts = []
        summary_parts.append(f"Based on {len(articles)} articles:")

        for article in articles[:5]:  # Top 5 articles
            title = article.get("title", "")[:100]
            year = article.get("year", "N/A")
            pmid = article.get("pmid", "")
            summary_parts.append(f"- {title} ({year}) [PMID: {pmid}]")

        return {
            "success": True,
            "synthesis": {
                "question": question,
                "article_count": len(articles),
                "study_types": study_types,
                "evidence_quality": quality,
                "summary": "\n".join(summary_parts),
                "citations": [
                    {"pmid": a.get("pmid"), "title": a.get("title"), "year": a.get("year")}
                    for a in articles
                ],
                "limitations": [
                    "This is an automated synthesis and should be reviewed",
                    f"Based on {len(articles)} articles which may not be comprehensive",
                ],
            },
        }

    except Exception as e:
        logger.error("Evidence synthesis failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "synthesis": None,
        }


# Create the tool for Google ADK
# Note: FunctionTool automatically extracts name from func.__name__ and description from func.__doc__
synthesize_evidence_tool = FunctionTool(func=synthesize_evidence)
