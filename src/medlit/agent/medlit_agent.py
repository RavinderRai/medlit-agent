"""MedLit Agent implementation using Google ADK."""

from __future__ import annotations

import asyncio
import os
import uuid

import structlog
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types

from medlit.config.constants import MEDICAL_DISCLAIMER
from medlit.config.settings import get_settings
from medlit.agent.tools.evidence_fetch import fetch_evidence_tool
from medlit.agent.tools.pubmed_search import search_pubmed_tool
from medlit.agent.tools.synthesize import synthesize_evidence_tool
from medlit.models import AgentResponse, ResponseStatus
from medlit.observability import MetricsTracker, init_langsmith
from medlit.observability.callbacks import MedLitCallbackHandler
from medlit.prompts import get_system_prompt

logger = structlog.get_logger(__name__)


class MedLitAgent:
    """Medical literature search and synthesis agent."""

    def __init__(
        self,
        model_name: str | None = None,
        enable_tracing: bool = True,
    ):
        """Initialize the MedLit agent.

        Args:
            model_name: Gemini model to use (defaults to settings)
            enable_tracing: Whether to enable LangSmith tracing
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.model_name

        # Set Google API key for ADK (it reads from environment)
        if self.settings.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.settings.google_api_key

        # Initialize observability FIRST (before creating agent)
        if enable_tracing:
            init_langsmith()

        self.metrics = MetricsTracker()
        self.callback_handler = MedLitCallbackHandler()

        # Create the ADK agent and runner
        self._agent = self._create_agent()
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app_name="medlit",
            agent=self._agent,
            session_service=self._session_service,
        )

        logger.info(
            "MedLit agent initialized",
            model=self.model_name,
            tracing_enabled=self.settings.langsmith_enabled,
        )

    def _create_agent(self) -> LlmAgent:
        """Create the Google ADK agent with tools."""
        system_prompt = get_system_prompt("agent_system")

        # Get the underlying functions from the FunctionTools
        tools = [
            search_pubmed_tool,
            fetch_evidence_tool,
            synthesize_evidence_tool,
        ]

        agent = LlmAgent(
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
            # Run the agent through ADK Runner
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
        """Run the agent through ADK Runner to process a question.

        Args:
            question: The medical question

        Returns:
            AgentResponse with results
        """
        # Create unique session for this query
        user_id = "medlit_user"
        session_id = str(uuid.uuid4())

        # Create session
        await self._session_service.create_session(
            app_name="medlit",
            user_id=user_id,
            session_id=session_id,
        )

        # Create the user message
        user_message = types.Content(
            parts=[types.Part(text=question)],
            role="user",
        )

        # Run the agent and collect events
        final_response = ""
        tool_calls = []

        try:
            # Use run_async for async execution
            # Collect all events to ensure generator is fully consumed
            events = []
            async for adk_event in self._runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_message,
            ):
                events.append(adk_event)

            # Process collected events
            for adk_event in events:
                # Log events for debugging
                logger.debug(
                    "ADK event received",
                    event_type=type(adk_event).__name__,
                    event_data=str(adk_event)[:200],
                )

                # Extract the final text response
                if hasattr(adk_event, 'content') and adk_event.content:
                    for part in adk_event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            final_response = part.text

                # Track tool calls
                if hasattr(adk_event, 'tool_calls'):
                    tool_calls.extend(adk_event.tool_calls)

        except Exception as e:
            logger.error("ADK runner error", error=str(e))
            raise

        # Build response
        if not final_response:
            return AgentResponse(
                question=question,
                status=ResponseStatus.NO_RESULTS,
                disclaimer=MEDICAL_DISCLAIMER,
            )

        return AgentResponse(
            question=question,
            status=ResponseStatus.SUCCESS,
            answer=final_response,
            disclaimer=MEDICAL_DISCLAIMER,
        )

    def ask_sync(self, question: str) -> AgentResponse:
        """Synchronous version of ask().

        Args:
            question: The medical question to answer

        Returns:
            AgentResponse with synthesized evidence
        """
        return asyncio.run(self.ask(question))


def create_agent(
    model_name: str | None = None,
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
