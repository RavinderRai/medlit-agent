"""Evidence synthesis tool for Google ADK agent."""

import json
import os

import structlog
from google import genai
from google.adk.tools import FunctionTool
from langsmith import traceable

from medlit.config.settings import get_settings
from medlit.prompts import get_template

logger = structlog.get_logger(__name__)


@traceable(name="synthesize_evidence", run_type="tool")
async def synthesize_evidence(
    question: str,
    articles_json: str,
) -> dict:
    """Synthesize evidence from multiple research articles using LLM.

    This tool uses an LLM to analyze and summarize the research articles,
    returning a concise synthesis instead of raw data.

    Args:
        question: The medical question being answered
        articles_json: JSON string containing list of article objects from fetch_evidence

    Returns:
        Dictionary with concise synthesis including key findings, evidence quality, and citations.
    """
    # Parse the JSON string
    try:
        parsed = json.loads(articles_json) if isinstance(articles_json, str) else articles_json
        # Handle both {"articles": [...]} and direct list formats
        if isinstance(parsed, dict):
            articles = parsed.get("articles", [])
        elif isinstance(parsed, list):
            articles = parsed
        else:
            articles = []
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
        # Format articles for the prompt (concise format to save tokens)
        articles_text = []
        for i, article in enumerate(articles[:10], 1):  # Limit to 10 articles
            pmid = article.get("pmid", "N/A")
            title = article.get("title", "No title")
            year = article.get("year", "N/A")
            article_type = article.get("article_type", "Unknown")
            abstract = article.get("abstract", "No abstract available")

            # Truncate abstract to save context
            if len(abstract) > 800:
                abstract = abstract[:800] + "..."

            articles_text.append(
                f"### Article {i} [PMID: {pmid}]\n"
                f"**Title**: {title}\n"
                f"**Year**: {year} | **Type**: {article_type}\n"
                f"**Abstract**: {abstract}\n"
            )

        formatted_articles = "\n---\n".join(articles_text)

        # Build the prompt from template
        template = get_template("evidence_synthesis_tool")
        prompt = template.format(
            question=question,
            articles_text=formatted_articles,
        )

        # Call Gemini to synthesize (using faster model for tools)
        settings = get_settings()
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model=settings.tool_model_name,
            contents=prompt,
        )

        # Parse the response
        response_text = response.text

        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                synthesis_data = json.loads(response_text[json_start:json_end])
            else:
                # Fallback if no JSON found
                synthesis_data = {
                    "key_finding": response_text[:500],
                    "evidence_summary": "",
                    "evidence_quality": "unknown",
                    "limitations": "",
                    "cited_pmids": [a.get("pmid") for a in articles[:5]],
                }
        except json.JSONDecodeError:
            synthesis_data = {
                "key_finding": response_text[:500],
                "evidence_summary": "",
                "evidence_quality": "unknown",
                "limitations": "",
                "cited_pmids": [a.get("pmid") for a in articles[:5]],
            }

        logger.info(
            "Evidence synthesis completed",
            evidence_quality=synthesis_data.get("evidence_quality"),
            cited_count=len(synthesis_data.get("cited_pmids", [])),
        )

        return {
            "success": True,
            "synthesis": {
                "question": question,
                "key_finding": synthesis_data.get("key_finding", ""),
                "evidence_summary": synthesis_data.get("evidence_summary", ""),
                "evidence_quality": synthesis_data.get("evidence_quality", "unknown"),
                "limitations": synthesis_data.get("limitations", ""),
                "cited_pmids": synthesis_data.get("cited_pmids", []),
                "articles_analyzed": len(articles),
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
synthesize_evidence_tool = FunctionTool(func=synthesize_evidence)
