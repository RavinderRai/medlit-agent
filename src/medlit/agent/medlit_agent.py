"""MedLit Agent implementation using Google ADK."""

import asyncio
from datetime import date
from typing import Optional

import structlog
from google import genai
from google.adk import Agent
from google.adk.tools import FunctionTool

from config.settings import get_settings
from config.constants import DEFAULT_DATE_RANGE_YEARS, MEDICAL_DISCLAIMER
from medlit.agent.tools.pubmed_search import search_pubmed_tool
from medlit.agent.tools.evidence_fetch import fetch_evidence_tool
from medlit.agent.tools.synthesize import synthesize_evidence_tool
from medlit.models import AgentResponse, ResponseStatus, SearchQuery, SearchFilters
from medlit.observability import init_langsmith, MetricsTracker
from medlit.observability.callbacks import MedLitCallbackHandler
from medlit.prompts import get_system_prompt
from medlit.pubmed import PubMedClient

logger = structlog.get_logger(__name__)


class MedLitAgent:
    """Medical literature search and synthesis agent."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        enable_tracing: bool = True,
    ):
        """Initialize the MedLit agent.

        Args:
            model_name: Gemini model to use (defaults to settings)
            enable_tracing: Whether to enable LangSmith tracing
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.model_name

        # Initialize observability
        if enable_tracing:
            init_langsmith()

        self.metrics = MetricsTracker()
        self.callback_handler = MedLitCallbackHandler()

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.settings.google_api_key)

        # Create the agent with tools
        self._agent = self._create_agent()

        logger.info(
            "MedLit agent initialized",
            model=self.model_name,
            tracing_enabled=self.settings.langsmith_enabled,
        )

    def _create_agent(self) -> Agent:
        """Create the Google ADK agent with tools."""
        system_prompt = get_system_prompt("agent_system")

        # Define tools
        tools = [
            search_pubmed_tool,
            fetch_evidence_tool,
            synthesize_evidence_tool,
        ]

        agent = Agent(
            model=self.model_name,
            name="medlit_agent",
            description="Medical literature search and synthesis agent",
            instruction=system_prompt,
            tools=tools,
        )

        return agent

    async def ask(self, question: str) -> AgentResponse:
        """Process a medical question and return synthesized evidence.

        Args:
            question: The medical question to answer

        Returns:
            AgentResponse with synthesized evidence and citations
        """
        # Start metrics tracking
        self.metrics.start_query(question)
        self.callback_handler.on_agent_start(question)

        try:
            # Run the agent
            response = await self._run_agent(question)

            self.callback_handler.on_agent_end(response)
            self.metrics.end_query(status="success")

            return response

        except Exception as e:
            logger.error("Agent error", error=str(e), question=question[:100])
            self.callback_handler.on_error(e, context="agent_ask")
            self.metrics.end_query(status="error")

            return AgentResponse(
                question=question,
                status=ResponseStatus.ERROR,
                error_message=str(e),
                disclaimer=MEDICAL_DISCLAIMER,
            )

    async def _run_agent(self, question: str) -> AgentResponse:
        """Run the agent to process a question.

        Args:
            question: The medical question

        Returns:
            AgentResponse with results
        """
        # Build default search query
        today = date.today()
        min_date = date(today.year - DEFAULT_DATE_RANGE_YEARS, today.month, today.day)

        search_query = SearchQuery(
            original_question=question,
            filters=SearchFilters(
                min_date=min_date,
                max_date=today,
            ),
        )

        # Search PubMed
        async with PubMedClient() as pubmed_client:
            # Generate optimized search query using LLM
            search_query = await self._generate_search_query(question, search_query)

            self.callback_handler.on_tool_start("pubmed_search", search_query.pubmed_query)

            # Execute search
            articles = await pubmed_client.search_and_fetch(search_query)

            self.callback_handler.on_tool_end("pubmed_search", f"{len(articles)} articles")

            self.metrics.record_search(
                pubmed_query=search_query.build_query(),
                articles_found=len(articles),
                latency_ms=0,  # Would need proper timing
            )

            if not articles:
                return AgentResponse(
                    question=question,
                    status=ResponseStatus.NO_RESULTS,
                    pubmed_query=search_query.build_query(),
                    articles_found=0,
                    articles_analyzed=0,
                    disclaimer=MEDICAL_DISCLAIMER,
                )

            # Synthesize evidence
            self.callback_handler.on_tool_start("synthesize", f"{len(articles)} articles")

            response = await self._synthesize_evidence(question, articles)

            self.callback_handler.on_tool_end("synthesize", "completed")

            return response

    async def _generate_search_query(
        self,
        question: str,
        base_query: SearchQuery,
    ) -> SearchQuery:
        """Use LLM to generate optimized PubMed search query.

        Args:
            question: Original question
            base_query: Base query with filters

        Returns:
            SearchQuery with optimized search terms
        """
        from medlit.prompts import get_template, get_few_shot_examples

        template = get_template("query_generation")
        examples = get_few_shot_examples("query_examples")

        # Format few-shot examples
        examples_text = "\n\n".join(
            f"Q: {ex['question']}\nA: {ex['output']}"
            for ex in examples[:3]
        )

        prompt = f"""
{template.format(question=question)}

## Examples
{examples_text}
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            # Parse response - in production, use structured output
            import json
            import re

            text = response.text
            # Extract JSON from response
            json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if json_match:
                query_data = json.loads(json_match.group())
                base_query.search_terms = query_data.get("search_terms", [])
                base_query.mesh_terms = query_data.get("mesh_terms", [])
                base_query.pubmed_query = query_data.get("pubmed_query", "")

        except Exception as e:
            logger.warning("Failed to generate optimized query", error=str(e))
            # Fall back to simple query
            base_query.pubmed_query = question

        return base_query

    async def _synthesize_evidence(
        self,
        question: str,
        articles: list,
    ) -> AgentResponse:
        """Synthesize evidence from articles.

        Args:
            question: Original question
            articles: List of Article objects

        Returns:
            AgentResponse with synthesized answer
        """
        from medlit.models import Citation, Evidence, EvidenceQuality
        from medlit.prompts import get_template

        # Format abstracts for LLM
        abstracts_text = "\n\n---\n\n".join(
            article.to_context_string() for article in articles
        )

        template = get_template("evidence_synthesis")
        prompt = template.format(
            question=question,
            abstracts=abstracts_text,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            # Parse synthesis response
            import json
            import re

            text = response.text
            json_match = re.search(r"\{[\s\S]*\}", text)

            if json_match:
                synthesis = json.loads(json_match.group())
            else:
                # Fallback to using raw text as summary
                synthesis = {"summary": text}

            # Build citations
            citations = [
                Citation(
                    pmid=article.pmid,
                    title=article.title,
                    authors=article.first_author + (" et al." if len(article.authors) > 1 else ""),
                    year=article.year,
                    journal=article.journal_abbrev or article.journal,
                )
                for article in articles
            ]

            # Build evidence object
            evidence = Evidence(
                summary=synthesis.get("summary", ""),
                quality=EvidenceQuality(synthesis.get("evidence_quality", "unknown")),
                supporting_citations=citations,
                limitations=synthesis.get("limitations", []),
                consensus=synthesis.get("consensus"),
            )

            self.metrics.record_result(
                citations_count=len(citations),
                evidence_quality=evidence.quality.value,
            )

            return AgentResponse(
                question=question,
                status=ResponseStatus.SUCCESS,
                answer=synthesis.get("summary", ""),
                evidence=evidence,
                citations=citations,
                pubmed_query="",  # Would be set from search
                articles_found=len(articles),
                articles_analyzed=len(articles),
                disclaimer=MEDICAL_DISCLAIMER,
            )

        except Exception as e:
            logger.error("Synthesis failed", error=str(e))
            raise

    def ask_sync(self, question: str) -> AgentResponse:
        """Synchronous version of ask().

        Args:
            question: The medical question to answer

        Returns:
            AgentResponse with synthesized evidence
        """
        return asyncio.run(self.ask(question))


def create_agent(
    model_name: Optional[str] = None,
    enable_tracing: bool = True,
) -> MedLitAgent:
    """Factory function to create a MedLit agent.

    Args:
        model_name: Gemini model to use
        enable_tracing: Whether to enable LangSmith tracing

    Returns:
        Configured MedLitAgent instance
    """
    return MedLitAgent(
        model_name=model_name,
        enable_tracing=enable_tracing,
    )
