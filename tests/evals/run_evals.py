"""Evaluation runner for MedLit Agent using LangSmith.

This script runs evaluations using LangSmith's evaluate() function
and displays results in the LangSmith dashboard.

Usage:
    uv run python -m tests.evals.run_evals
    uv run python -m tests.evals.run_evals --quick  # Only programmatic evals
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import structlog
from langsmith import Client, evaluate
from langsmith.schemas import Example, Run

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medlit.agent import create_agent
from medlit.observability import init_langsmith

from tests.evals.evaluators import (
    ALL_EVALUATORS,
    LLM_JUDGE_EVALUATORS,
    PROGRAMMATIC_EVALUATORS,
    answer_quality,
    citation_accuracy,
    faithfulness,
    has_citations,
    relevance,
    tool_trace_valid,
    topic_coverage,
)

logger = structlog.get_logger(__name__)

DATASET_PATH = Path(__file__).parent / "datasets" / "medical_qa.json"
DATASET_NAME = "medlit-medical-qa"


def load_local_dataset() -> list[dict]:
    """Load the evaluation dataset from local JSON file."""
    with open(DATASET_PATH) as f:
        return json.load(f)


def create_or_update_langsmith_dataset(client: Client) -> str:
    """Create or update the LangSmith dataset from local JSON.

    Returns:
        Dataset ID
    """
    local_data = load_local_dataset()

    # Check if dataset exists
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        logger.info("Using existing dataset", dataset_id=dataset.id)
    except Exception:
        # Create new dataset
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Medical Q&A evaluation dataset for MedLit Agent",
        )
        logger.info("Created new dataset", dataset_id=dataset.id)

        # Add examples
        for item in local_data:
            client.create_example(
                inputs={"question": item["question"]},
                outputs={
                    "expected_topics": item.get("expected_topics", []),
                    "expected_conclusion": item.get("expected_conclusion", ""),
                    "difficulty": item.get("difficulty", "medium"),
                },
                dataset_id=dataset.id,
                metadata={"id": item["id"]},
            )
        logger.info("Added examples to dataset", count=len(local_data))

    return dataset.id


# Global agent instance for reuse
_agent = None


def get_agent():
    """Get or create the agent instance."""
    global _agent
    if _agent is None:
        _agent = create_agent(enable_tracing=True)
    return _agent


def target_function(inputs: dict) -> dict:
    """Target function that runs the agent on a question.

    This is called by LangSmith's evaluate() for each example.
    """
    question = inputs["question"]
    agent = get_agent()

    # Run the agent
    response = asyncio.run(agent.ask(question))

    # Convert response to dict for evaluators
    return {
        "answer": response.answer or "",
        "status": response.status.value,
        "citations": [
            {
                "pmid": c.pmid,
                "title": c.title,
                "authors": c.authors,
                "year": c.year,
            }
            for c in (response.citations or [])
        ],
        "evidence_quality": (
            response.evidence.quality.value if response.evidence else None
        ),
        "articles_found": response.articles_found or 0,
        "error": response.error_message,
    }


def create_evaluator_wrapper(eval_func):
    """Wrap our evaluator functions to work with LangSmith's expected signature."""

    def wrapper(run: Run, example: Example) -> dict:
        outputs = run.outputs or {}
        reference_outputs = example.outputs or {}
        # Add question from inputs for context
        reference_outputs["question"] = example.inputs.get("question", "")

        result = eval_func(outputs, reference_outputs)
        return {
            "key": result["key"],
            "score": result["score"],
            "comment": result.get("reasoning", ""),
        }

    wrapper.__name__ = eval_func.__name__
    return wrapper


def run_evaluation(quick: bool = False, experiment_prefix: str = "medlit-eval"):
    """Run the full evaluation suite.

    Args:
        quick: If True, only run programmatic evaluators (faster, no LLM calls)
        experiment_prefix: Prefix for the experiment name in LangSmith
    """
    # Initialize LangSmith
    init_langsmith()

    client = Client()

    # Create or update dataset
    logger.info("Setting up dataset...")
    dataset_id = create_or_update_langsmith_dataset(client)

    # Select evaluators
    if quick:
        evaluators = PROGRAMMATIC_EVALUATORS
        logger.info("Running quick evaluation (programmatic only)")
    else:
        evaluators = ALL_EVALUATORS
        logger.info("Running full evaluation (including LLM judges)")

    # Wrap evaluators for LangSmith
    wrapped_evaluators = [create_evaluator_wrapper(e) for e in evaluators]

    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{experiment_prefix}-{timestamp}"

    logger.info(
        "Starting evaluation",
        experiment=experiment_name,
        evaluators=[e.__name__ for e in evaluators],
    )

    # Run evaluation
    results = evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=wrapped_evaluators,
        experiment_prefix=experiment_prefix,
        metadata={
            "quick": quick,
            "timestamp": timestamp,
        },
    )

    # Print summary
    print_results_summary(results, evaluators)

    return results


def print_results_summary(results, evaluators):
    """Print a summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Collect scores by evaluator
    scores_by_eval = {e.__name__: [] for e in evaluators}

    for result in results:
        if hasattr(result, "evaluation_results") and result.evaluation_results:
            for eval_result in result.evaluation_results:
                key = eval_result.key
                if key in scores_by_eval:
                    scores_by_eval[key].append(eval_result.score or 0)

    # Print per-evaluator results
    print("\nPer-Evaluator Results:")
    print("-" * 60)

    all_pass_rates = []
    for eval_name, scores in scores_by_eval.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            pass_rate = sum(1 for s in scores if s >= 0.5) / len(scores)
            all_pass_rates.append(pass_rate)

            # Create visual bar
            bar_len = 20
            filled = int(pass_rate * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            print(f"{eval_name:20} {bar} {pass_rate:6.1%} ({sum(scores):.1f}/{len(scores)})")

    # Overall pass rate
    if all_pass_rates:
        overall = sum(all_pass_rates) / len(all_pass_rates)
        print("-" * 60)
        print(f"{'OVERALL':20} {'':20} {overall:6.1%}")

    print("\n" + "=" * 60)
    print("View detailed results at: https://smith.langchain.com")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run MedLit Agent evaluations")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only run programmatic evaluators (faster, no LLM judge calls)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="medlit-eval",
        help="Experiment name prefix",
    )

    args = parser.parse_args()

    # Check for required env vars
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("Error: LANGSMITH_API_KEY environment variable not set")
        print("Set it in your .env file or export it")
        sys.exit(1)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        sys.exit(1)

    try:
        run_evaluation(quick=args.quick, experiment_prefix=args.prefix)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
