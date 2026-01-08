#!/usr/bin/env python3
"""Quick run script for MedLit Agent.

Usage:
    python scripts/run_agent.py "Your medical question here"
    python scripts/run_agent.py --interactive
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medlit.agent import create_agent
from medlit.utils.validators import validate_question, ValidationError


async def ask_question(question: str) -> None:
    """Ask a single question and print the response."""
    print(f"\n🔍 Question: {question}\n")
    print("Searching PubMed and synthesizing evidence...\n")

    agent = create_agent(enable_tracing=False)
    response = await agent.ask(question)

    print("=" * 60)
    print(response.to_markdown())
    print("=" * 60)


async def interactive_mode() -> None:
    """Run in interactive mode."""
    print("\n" + "=" * 60)
    print("MedLit Agent - Interactive Mode")
    print("=" * 60)
    print("Ask medical questions and get evidence-based answers.")
    print("Type 'quit' or 'exit' to stop.\n")

    agent = create_agent(enable_tracing=False)

    while True:
        try:
            question = input("\n📝 Your question: ").strip()

            if question.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            if not question:
                continue

            try:
                question = validate_question(question)
            except ValidationError as e:
                print(f"❌ {e.message}")
                continue

            print("\n🔍 Searching...")
            response = await agent.ask(question)

            print("\n" + "-" * 60)
            print(response.to_markdown())
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MedLit Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_agent.py "Is aspirin good for heart health?"
    python scripts/run_agent.py --interactive
        """,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="The medical question to ask",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode",
    )

    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_mode())
    elif args.question:
        asyncio.run(ask_question(args.question))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
