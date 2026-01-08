"""Tests for prompt registry."""

import pytest
from pathlib import Path

from medlit.prompts import PromptRegistry, get_system_prompt, get_template, get_few_shot_examples


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    @pytest.fixture
    def registry(self):
        """Create registry with test prompts directory."""
        return PromptRegistry()

    def test_load_system_prompt(self, registry):
        """Test loading system prompt."""
        prompt = registry.load("system", "agent_system")
        assert "MedLit" in prompt
        assert len(prompt) > 100

    def test_load_template(self, registry):
        """Test loading template."""
        template = registry.load("templates", "query_generation")
        assert "{question}" in template

    def test_load_missing_prompt(self, registry):
        """Test loading non-existent prompt raises error."""
        with pytest.raises(FileNotFoundError):
            registry.load("system", "nonexistent")

    def test_cache_works(self, registry):
        """Test that caching works."""
        # Load twice
        prompt1 = registry.load("system", "agent_system")
        prompt2 = registry.load("system", "agent_system")
        assert prompt1 == prompt2

    def test_format_prompt(self, registry):
        """Test prompt formatting."""
        formatted = registry.format_prompt(
            "templates",
            "query_generation",
            question="What is aspirin?",
        )
        assert "What is aspirin?" in formatted

    def test_load_json(self, registry):
        """Test loading JSON file."""
        examples = registry.load_json("few_shot", "query_examples")
        assert isinstance(examples, list)
        assert len(examples) > 0
        assert "question" in examples[0]


class TestPromptHelpers:
    """Tests for prompt helper functions."""

    def test_get_system_prompt(self):
        """Test get_system_prompt helper."""
        prompt = get_system_prompt("agent_system")
        assert "MedLit" in prompt

    def test_get_template(self):
        """Test get_template helper."""
        template = get_template("evidence_synthesis")
        assert "{question}" in template
        assert "{abstracts}" in template

    def test_get_few_shot_examples(self):
        """Test get_few_shot_examples helper."""
        examples = get_few_shot_examples("query_examples")
        assert isinstance(examples, list)
        assert all("question" in ex for ex in examples)
