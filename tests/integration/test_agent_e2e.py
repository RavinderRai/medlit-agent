"""End-to-end tests for MedLit Agent.

These tests require API keys and are skipped in CI.
"""

import pytest


pytestmark = pytest.mark.skipif(
    True,  # Set to False to run live tests
    reason="E2E tests disabled by default - require API keys",
)


class TestMedLitAgentE2E:
    """End-to-end tests for MedLit agent."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        from medlit.agent import create_agent

        return create_agent(enable_tracing=False)

    @pytest.mark.asyncio
    async def test_ask_medical_question(self, agent):
        """Test asking a medical question."""
        question = "What are the common side effects of metformin?"

        response = await agent.ask(question)

        assert response.question == question
        assert response.status.value in ["success", "no_results"]

        if response.status.value == "success":
            assert response.answer
            assert len(response.citations) > 0

    @pytest.mark.asyncio
    async def test_ask_returns_citations(self, agent):
        """Test that response includes citations."""
        question = "Is aspirin effective for primary prevention of cardiovascular disease?"

        response = await agent.ask(question)

        if response.status.value == "success":
            assert len(response.citations) > 0
            for citation in response.citations:
                assert citation.pmid
                assert citation.title

    @pytest.mark.asyncio
    async def test_ask_includes_disclaimer(self, agent):
        """Test that response includes medical disclaimer."""
        question = "What is the recommended treatment for hypertension?"

        response = await agent.ask(question)

        assert response.disclaimer
        assert "not a substitute" in response.disclaimer.lower() or "healthcare" in response.disclaimer.lower()

    def test_ask_sync(self, agent):
        """Test synchronous ask method."""
        question = "What are the benefits of exercise for depression?"

        response = agent.ask_sync(question)

        assert response.question == question
        assert response.status.value in ["success", "no_results", "error"]
