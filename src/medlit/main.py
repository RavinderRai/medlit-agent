"""MedLit Agent CLI entrypoint."""

import asyncio
import sys
from typing import Optional

import click
import structlog

from medlit.config.settings import get_settings


def setup_logging(log_level: str) -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if log_level == "DEBUG" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default=None,
    help="Set logging level",
)
@click.pass_context
def cli(ctx: click.Context, log_level: Optional[str]) -> None:
    """MedLit Agent - Medical literature search and synthesis."""
    ctx.ensure_object(dict)

    settings = get_settings()
    effective_log_level = log_level or settings.log_level

    setup_logging(effective_log_level)
    ctx.obj["settings"] = settings


@cli.command()
@click.argument("question")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "markdown", "json"]),
    default="markdown",
    help="Output format",
)
@click.option(
    "--max-results",
    type=int,
    default=8,
    help="Maximum number of articles to analyze",
)
@click.option(
    "--no-tracing",
    is_flag=True,
    help="Disable LangSmith tracing",
)
@click.pass_context
def ask(
    ctx: click.Context,
    question: str,
    output_format: str,
    max_results: int,
    no_tracing: bool,
) -> None:
    """Ask a medical question and get evidence-based answer.

    Example:
        medlit ask "Is low-dose aspirin recommended for preventing heart attacks?"
    """
    from medlit.agent import create_agent
    from medlit.utils.validators import validate_question, ValidationError

    # Validate input
    try:
        question = validate_question(question)
    except ValidationError as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)

    click.echo(f"Searching for: {question}\n")
    click.echo("This may take a moment...\n")

    # Create and run agent
    agent = create_agent(enable_tracing=not no_tracing)

    try:
        response = asyncio.run(agent.ask(question))

        # Output based on format
        if output_format == "json":
            click.echo(response.model_dump_json(indent=2))
        elif output_format == "markdown":
            click.echo(response.to_markdown())
        else:
            click.echo(response.to_text())

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option(
    "--max-results",
    type=int,
    default=10,
    help="Maximum number of results",
)
@click.option(
    "--years",
    type=int,
    default=5,
    help="Years to search back",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    max_results: int,
    years: int,
) -> None:
    """Search PubMed directly without synthesis.

    Example:
        medlit search "aspirin[MeSH] AND cardiovascular[MeSH]"
    """
    from datetime import date
    from medlit.models import SearchQuery, SearchFilters
    from medlit.pubmed import PubMedClient

    click.echo(f"Searching PubMed: {query}\n")

    today = date.today()
    min_date = date(today.year - years, today.month, today.day)

    search_query = SearchQuery(
        original_question=query,
        pubmed_query=query,
        filters=SearchFilters(min_date=min_date, max_date=today),
        max_results=max_results,
    )

    async def run_search():
        async with PubMedClient() as client:
            articles = await client.search_and_fetch(search_query)
            return articles

    try:
        articles = asyncio.run(run_search())

        if not articles:
            click.echo("No results found.")
            return

        click.echo(f"Found {len(articles)} articles:\n")

        for i, article in enumerate(articles, 1):
            click.echo(f"{i}. {article.title}")
            click.echo(f"   Authors: {article.first_author}" + (" et al." if len(article.authors) > 1 else ""))
            click.echo(f"   Journal: {article.journal} ({article.year})")
            click.echo(f"   PMID: {article.pmid}")
            click.echo(f"   URL: {article.pubmed_url}")
            if article.article_type:
                click.echo(f"   Type: {article.article_type}")
            click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Show current configuration."""
    settings = ctx.obj["settings"]

    click.echo("MedLit Agent Configuration")
    click.echo("=" * 40)
    click.echo(f"Environment: {settings.environment}")
    click.echo(f"Log Level: {settings.log_level}")
    click.echo(f"Model: {settings.model_name}")
    click.echo()
    click.echo("API Keys:")
    click.echo(f"  Google API: {'✓ Set' if settings.google_api_key else '✗ Not set'}")
    click.echo(f"  LangSmith: {'✓ Set' if settings.langchain_api_key else '✗ Not set'}")
    click.echo(f"  NCBI/PubMed: {'✓ Set' if settings.ncbi_api_key else '✗ Not set (using default rate limit)'}")
    click.echo()
    click.echo("Features:")
    click.echo(f"  LangSmith Tracing: {'Enabled' if settings.langsmith_enabled else 'Disabled'}")
    click.echo(f"  Redis Cache: {'Configured' if settings.has_redis else 'Not configured (using in-memory)'}")


@cli.command()
def version() -> None:
    """Show version information."""
    from medlit import __version__

    click.echo(f"MedLit Agent v{__version__}")


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
