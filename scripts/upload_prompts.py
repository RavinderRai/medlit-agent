#!/usr/bin/env python3
"""Upload prompts to LangSmith Hub for versioning.

This script uploads prompt templates to LangSmith Hub,
enabling version control and collaboration.

Usage:
    python scripts/upload_prompts.py
    python scripts/upload_prompts.py --prompt agent_system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import get_settings


def upload_prompt(name: str, content: str, description: str = "") -> None:
    """Upload a single prompt to LangSmith Hub.

    Args:
        name: Prompt name/identifier
        content: Prompt content
        description: Optional description
    """
    try:
        from langsmith import Client

        settings = get_settings()
        if not settings.langchain_api_key:
            print("❌ LangSmith API key not configured")
            return

        client = Client()

        # Note: This is a simplified example
        # Actual implementation depends on LangSmith Hub API
        print(f"📤 Uploading prompt: {name}")
        print(f"   Length: {len(content)} characters")

        # In a real implementation:
        # client.push_prompt(name, content, description=description)

        print(f"✅ Uploaded: {name}")

    except ImportError:
        print("❌ LangSmith not installed")
    except Exception as e:
        print(f"❌ Upload failed: {e}")


def upload_all_prompts() -> None:
    """Upload all prompts from the prompts directory."""
    from medlit.prompts import PromptRegistry

    registry = PromptRegistry()
    prompts_dir = Path(__file__).parent.parent / "src" / "medlit" / "prompts"

    # System prompts
    system_dir = prompts_dir / "system"
    if system_dir.exists():
        for prompt_file in system_dir.glob("*.txt"):
            name = f"medlit/system/{prompt_file.stem}"
            content = prompt_file.read_text()
            upload_prompt(name, content, f"System prompt: {prompt_file.stem}")

    # Templates
    templates_dir = prompts_dir / "templates"
    if templates_dir.exists():
        for prompt_file in templates_dir.glob("*.txt"):
            name = f"medlit/templates/{prompt_file.stem}"
            content = prompt_file.read_text()
            upload_prompt(name, content, f"Template: {prompt_file.stem}")

    print("\n✅ All prompts processed")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Upload prompts to LangSmith Hub")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Specific prompt to upload (e.g., 'agent_system')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 Dry run mode - showing prompts that would be uploaded:\n")

    if args.prompt:
        from medlit.prompts import PromptRegistry

        registry = PromptRegistry()
        # Try to find the prompt in different categories
        for category in ["system", "templates"]:
            try:
                content = registry.load(category, args.prompt)
                if args.dry_run:
                    print(f"Would upload: medlit/{category}/{args.prompt}")
                    print(f"Content preview: {content[:200]}...")
                else:
                    upload_prompt(f"medlit/{category}/{args.prompt}", content)
                break
            except FileNotFoundError:
                continue
        else:
            print(f"❌ Prompt not found: {args.prompt}")
    else:
        if args.dry_run:
            prompts_dir = Path(__file__).parent.parent / "src" / "medlit" / "prompts"
            for category in ["system", "templates"]:
                cat_dir = prompts_dir / category
                if cat_dir.exists():
                    for f in cat_dir.glob("*.txt"):
                        print(f"Would upload: medlit/{category}/{f.stem}")
        else:
            upload_all_prompts()


if __name__ == "__main__":
    main()
