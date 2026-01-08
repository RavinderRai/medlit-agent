from functools import lru_cache
from typing import Literal

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Google ADK / Gemini
    google_api_key: str = Field(
        default="",
        description="Google API key for Gemini models",
    )

    # LangSmith observability
    # Uses validation_alias to read from LANGCHAIN_* env vars (LangSmith's official naming)
    langsmith_tracing: bool = Field(
        default=False,
        validation_alias=AliasChoices("langsmith_tracing", "langchain_tracing_v2"),
        description="Enable LangSmith tracing",
    )
    langsmith_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("langsmith_api_key", "langchain_api_key"),
        description="LangSmith API key",
    )
    langsmith_project: str = Field(
        default="medlit-agent",
        validation_alias=AliasChoices("langsmith_project", "langchain_project"),
        description="LangSmith project name",
    )

    # PubMed
    ncbi_api_key: str = Field(
        default="",
        description="NCBI API key for increased rate limits",
    )

    # Cache
    redis_url: str = Field(
        default="",
        description="Redis URL for caching",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Model settings
    model_name: str = Field(
        default="gemini-2.0-flash",
        description="Default Gemini model to use",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for LLM response",
    )
    temperature: float = Field(
        default=0.1,
        description="Temperature for LLM response (lower = more deterministic)",
    )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def has_ncbi_key(self) -> bool:
        return bool(self.ncbi_api_key)

    @property
    def has_redis(self) -> bool:
        return bool(self.redis_url)

    @property
    def langsmith_enabled(self) -> bool:
        return self.langsmith_tracing and bool(self.langsmith_api_key)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
