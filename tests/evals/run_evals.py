"""Evaluation runner for MedLit Agent.

This script runs evaluations using the test dataset and optionally
reports results to LangSmith.

Usage:
    python -m tests.evals.run_evals [--langsmith] [--output results.json]
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

DATASET_PATH = Path(__file__).parent / "datasets" / "medical_qa.json"


def load_dataset() -> list[dict]:
    """Load the evaluation dataset."""
    with open(DATASET_PATH) as f:
        return json.load(f)


def evaluate_response(response, expected: dict) -> dict:
    """Evaluate a single response against expected criteria.

    Args:
        response: AgentResponse object
        expected: Expected criteria from dataset

    Returns:
        Evaluation results
    """
    results = {
        "question_id": expected["id"],
        "status": response.status.value,
        "has_answer": bool(response.answer),
        "has_citations": len(response.citations) > 0,
        "citation_count": len(response.citations),
        "topics_covered": [],
        "topics_missing": [],
        "score": 0.0,
    }

    if response.status.value != "success":
        return results

    # Check if expected topics are covered in the answer
    answer_lower = response.answer.lower()
    expected_topics = expected.get("expected_topics", [])

    for topic in expected_topics:
        # Simple keyword matching - could be improved with embeddings
        topic_words = topic.lower().split()
        if any(word in answer_lower for word in topic_words):
            results["topics_covered"].append(topic)
        else:
            results["topics_missing"].append(topic)

    # Calculate score
    if expected_topics:
        topic_coverage = len(results["topics_covered"]) / len(expected_topics)
    else:
        topic_coverage = 1.0

    citation_score = min(1.0, len(response.citations) / 3)  # Target 3+ citations
    has_evidence = 1.0 if response.evidence else 0.0

    results["score"] = (topic_coverage * 0.4 + citation_score * 0.3 + has_evidence * 0.3)

    return results


async def run_evaluation(
    use_langsmith: bool = False,
    output_file: Optional[str] = None,
) -> dict:
    """Run full evaluation suite.

    Args:
        use_langsmith: Whether to report to LangSmith
        output_file: Optional file to write results

    Returns:
        Evaluation summary
    """
    from medlit.agent import create_agent

    dataset = load_dataset()
    logger.info("Loaded evaluation dataset", count=len(dataset))

    agent = create_agent(enable_tracing=use_langsmith)

    results = []
    for item in dataset:
        logger.info("Evaluating", question_id=item["id"])

        try:
            response = await agent.ask(item["question"])
            eval_result = evaluate_response(response, item)
            eval_result["error"] = None
        except Exception as e:
            logger.error("Evaluation failed", question_id=item["id"], error=str(e))
            eval_result = {
                "question_id": item["id"],
                "status": "error",
                "error": str(e),
                "score": 0.0,
            }

        results.append(eval_result)

    # Calculate summary statistics
    successful = [r for r in results if r["status"] == "success"]
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_questions": len(dataset),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "average_score": sum(r["score"] for r in results) / len(results) if results else 0,
        "average_citations": (
            sum(r.get("citation_count", 0) for r in successful) / len(successful)
            if successful else 0
        ),
        "results": results,
    }

    logger.info(
        "Evaluation complete",
        total=summary["total_questions"],
        successful=summary["successful"],
        avg_score=f"{summary['average_score']:.2f}",
    )

    # Write results to file if specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results written", file=output_file)

    return summary


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run MedLit Agent evaluations")
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Report results to LangSmith",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    summary = asyncio.run(run_evaluation(
        use_langsmith=args.langsmith,
        output_file=args.output,
    ))

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total questions: {summary['total_questions']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Average score: {summary['average_score']:.2%}")
    print(f"Average citations: {summary['average_citations']:.1f}")

    # Exit with error if too many failures
    if summary["failed"] > summary["total_questions"] / 2:
        sys.exit(1)


if __name__ == "__main__":
    main()
