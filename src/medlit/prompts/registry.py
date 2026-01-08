"""Prompt loading and versioning registry."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent


class PromptRegistry:
    """Registry for managing and loading prompts."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self._cache: dict[str, str] = {}

    def load(self, category: str, name: str, use_cache: bool = True) -> str:
        """Load a prompt from file.

        Args:
            category: Prompt category (system, templates, etc.)
            name: Prompt name (without extension)
            use_cache: Whether to use cached version

        Returns:
            Prompt content as string
        """
        cache_key = f"{category}/{name}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        path = self.prompts_dir / category / f"{name}.txt"

        if not path.exists():
            logger.error("Prompt not found", category=category, name=name, path=str(path))
            raise FileNotFoundError(f"Prompt not found: {path}")

        content = path.read_text(encoding="utf-8").strip()

        if use_cache:
            self._cache[cache_key] = content

        logger.debug("Loaded prompt", category=category, name=name)
        return content

    def load_json(self, category: str, name: str) -> Any:
        """Load a JSON file (e.g., few-shot examples).

        Args:
            category: Prompt category
            name: File name (without extension)

        Returns:
            Parsed JSON content
        """
        path = self.prompts_dir / category / f"{name}.json"

        if not path.exists():
            logger.error("JSON file not found", category=category, name=name, path=str(path))
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def format_prompt(
        self,
        category: str,
        name: str,
        **kwargs: Any,
    ) -> str:
        """Load and format a prompt template.

        Args:
            category: Prompt category
            name: Prompt name
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string
        """
        template = self.load(category, name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error("Missing template variable", category=category, name=name, missing=str(e))
            raise

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()
        logger.debug("Prompt cache cleared")


# Global registry instance
_registry = PromptRegistry()


@lru_cache(maxsize=32)
def load_prompt(category: str, name: str) -> str:
    """Load a prompt from file with caching."""
    return _registry.load(category, name)


def get_system_prompt(name: str) -> str:
    """Load a system prompt."""
    return load_prompt("system", name)


def get_template(name: str) -> str:
    """Load a prompt template."""
    return load_prompt("templates", name)


def get_few_shot_examples(name: str) -> list[dict[str, Any]]:
    """Load few-shot examples from JSON."""
    return _registry.load_json("few_shot", name)
