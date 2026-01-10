"""Evaluators for MedLit Agent using LangSmith.

This module defines evaluators that can be used with LangSmith's evaluate() function.
Each evaluator returns a score and optional feedback.
"""

import os
import re

from dotenv import load_dotenv
from google import genai

load_dotenv()

# ============================================================================
# Programmatic Evaluators
# ============================================================================


def citation_accuracy(outputs: dict, reference_outputs: dict = None) -> dict:
    """Check if PMIDs mentioned in the answer are valid and were actually fetched.

    Returns:
        dict with 'score' (0 or 1) and 'reasoning'
    """
    answer = outputs.get("answer", "")
    citations = outputs.get("citations", [])

    if not answer:
        return {
            "key": "citation_accuracy",
            "score": 0,
            "reasoning": "No answer provided",
        }

    # Extract PMIDs mentioned in the answer text
    # PMIDs are typically 1-8 digit numbers, often in format [PMID: 12345678] or (PMID: 12345678)
    pmid_patterns = [
        r"PMID:\s*(\d{1,8})",
        r"PMID\s+(\d{1,8})",
        r"\[(\d{7,8})\]",  # [12345678]
        r"\((\d{7,8})\)",  # (12345678)
    ]

    mentioned_pmids = set()
    for pattern in pmid_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        mentioned_pmids.update(matches)

    if not mentioned_pmids:
        # No PMIDs mentioned - could be okay if answer doesn't cite specific studies
        return {
            "key": "citation_accuracy",
            "score": 1,
            "reasoning": "No specific PMIDs mentioned in answer (acceptable)",
        }

    # Get PMIDs that were actually fetched
    fetched_pmids = set()
    for citation in citations:
        if hasattr(citation, "pmid"):
            fetched_pmids.add(str(citation.pmid))
        elif isinstance(citation, dict):
            fetched_pmids.add(str(citation.get("pmid", "")))

    # Check if all mentioned PMIDs were in the fetched set
    invalid_pmids = mentioned_pmids - fetched_pmids

    if invalid_pmids:
        return {
            "key": "citation_accuracy",
            "score": 0,
            "reasoning": f"Hallucinated PMIDs found: {invalid_pmids}. Fetched: {fetched_pmids}",
        }

    return {
        "key": "citation_accuracy",
        "score": 1,
        "reasoning": f"All {len(mentioned_pmids)} mentioned PMIDs are valid",
    }


def tool_trace_valid(outputs: dict, reference_outputs: dict = None) -> dict:
    """Check if the agent used tools correctly.

    For MedLit, we expect: search_pubmed -> fetch_evidence -> synthesize_evidence

    Returns:
        dict with 'score' (0 or 1) and 'reasoning'
    """
    status = outputs.get("status", "")
    citations = outputs.get("citations", [])
    answer = outputs.get("answer", "")

    # Check for success indicators
    checks = {
        "has_answer": bool(answer),
        "has_citations": len(citations) > 0,
        "status_success": status == "success",
    }

    passed = sum(checks.values())
    total = len(checks)

    if passed == total:
        return {
            "key": "tool_trace_valid",
            "score": 1,
            "reasoning": "All tool trace checks passed: answer generated with citations",
        }
    else:
        failed = [k for k, v in checks.items() if not v]
        return {
            "key": "tool_trace_valid",
            "score": 0,
            "reasoning": f"Tool trace incomplete. Failed checks: {failed}",
        }


def has_citations(outputs: dict, reference_outputs: dict = None) -> dict:
    """Simple check: does the response have any citations?

    Returns:
        dict with 'score' (0 or 1) and 'reasoning'
    """
    citations = outputs.get("citations", [])

    if len(citations) >= 3:
        return {
            "key": "has_citations",
            "score": 1,
            "reasoning": f"Has {len(citations)} citations (target: 3+)",
        }
    elif len(citations) > 0:
        return {
            "key": "has_citations",
            "score": 0.5,
            "reasoning": f"Has only {len(citations)} citations (target: 3+)",
        }
    else:
        return {
            "key": "has_citations",
            "score": 0,
            "reasoning": "No citations provided",
        }


def topic_coverage(outputs: dict, reference_outputs: dict = None) -> dict:
    """Check if expected topics are covered in the answer.

    Returns:
        dict with 'score' (0-1) and 'reasoning'
    """
    answer = outputs.get("answer", "").lower()
    expected_topics = reference_outputs.get("expected_topics", []) if reference_outputs else []

    if not expected_topics:
        return {
            "key": "topic_coverage",
            "score": 1,
            "reasoning": "No expected topics defined",
        }

    covered = []
    missing = []

    for topic in expected_topics:
        topic_words = topic.lower().split()
        if any(word in answer for word in topic_words):
            covered.append(topic)
        else:
            missing.append(topic)

    score = len(covered) / len(expected_topics)

    return {
        "key": "topic_coverage",
        "score": score,
        "reasoning": f"Covered {len(covered)}/{len(expected_topics)} topics. Missing: {missing}",
    }


# ============================================================================
# LLM-as-Judge Evaluators
# ============================================================================


def _call_llm_judge(prompt: str) -> str:
    """Call Gemini to evaluate."""
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def faithfulness(outputs: dict, reference_outputs: dict = None) -> dict:
    """LLM judge: Is the answer grounded in the provided sources?

    Checks for hallucination - claims not supported by the retrieved evidence.

    Returns:
        dict with 'score' (0 or 1) and 'reasoning'
    """
    answer = outputs.get("answer", "")
    citations = outputs.get("citations", [])

    if not answer:
        return {
            "key": "faithfulness",
            "score": 0,
            "reasoning": "No answer provided",
        }

    # Format citations for context
    citation_text = ""
    for c in citations[:5]:
        if hasattr(c, "title"):
            citation_text += f"- PMID {c.pmid}: {c.title}\n"
        elif isinstance(c, dict):
            citation_text += f"- PMID {c.get('pmid')}: {c.get('title', 'N/A')}\n"

    prompt = f"""You are evaluating whether a medical answer is faithful to its sources.

SOURCES (article titles):
{citation_text if citation_text else "No sources provided"}

ANSWER:
{answer[:2000]}

TASK: Does this answer only make claims that could reasonably be supported by the listed sources?
- A faithful answer only states information that appears to come from the cited sources
- An unfaithful answer makes up facts, cites non-existent studies, or claims things not in the sources

Respond with ONLY:
SCORE: 1 (if faithful) or 0 (if unfaithful)
REASON: One sentence explanation
"""

    try:
        response = _call_llm_judge(prompt)

        # Parse response
        score = 1 if "SCORE: 1" in response or "SCORE:1" in response else 0
        reason_match = re.search(r"REASON:\s*(.+)", response, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else response[:200]

        return {
            "key": "faithfulness",
            "score": score,
            "reasoning": reason,
        }
    except Exception as e:
        return {
            "key": "faithfulness",
            "score": 0,
            "reasoning": f"Evaluation failed: {str(e)}",
        }


def answer_quality(outputs: dict, reference_outputs: dict = None) -> dict:
    """LLM judge: Overall quality of the answer.

    Evaluates clarity, completeness, and medical accuracy.

    Returns:
        dict with 'score' (0-1) and 'reasoning'
    """
    question = reference_outputs.get("question", "") if reference_outputs else ""
    answer = outputs.get("answer", "")

    if not answer:
        return {
            "key": "answer_quality",
            "score": 0,
            "reasoning": "No answer provided",
        }

    prompt = f"""You are evaluating the quality of a medical literature answer.

QUESTION: {question}

ANSWER:
{answer[:2000]}

TASK: Rate the overall quality of this answer on a scale of 0-10 based on:
- Clarity: Is it well-written and easy to understand?
- Completeness: Does it address the question fully?
- Medical accuracy: Does it seem medically sound (not giving dangerous advice)?
- Evidence-based: Does it reference studies/evidence?

Respond with ONLY:
SCORE: [0-10]
REASON: One sentence explanation
"""

    try:
        response = _call_llm_judge(prompt)

        # Parse response
        score_match = re.search(r"SCORE:\s*(\d+)", response)
        score = int(score_match.group(1)) / 10 if score_match else 0.5

        reason_match = re.search(r"REASON:\s*(.+)", response, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else response[:200]

        return {
            "key": "answer_quality",
            "score": min(1.0, max(0.0, score)),  # Clamp to 0-1
            "reasoning": reason,
        }
    except Exception as e:
        return {
            "key": "answer_quality",
            "score": 0,
            "reasoning": f"Evaluation failed: {str(e)}",
        }


def relevance(outputs: dict, reference_outputs: dict = None) -> dict:
    """LLM judge: Are the retrieved sources relevant to the question?

    Returns:
        dict with 'score' (0 or 1) and 'reasoning'
    """
    question = reference_outputs.get("question", "") if reference_outputs else ""
    citations = outputs.get("citations", [])

    if not citations:
        return {
            "key": "relevance",
            "score": 0,
            "reasoning": "No sources retrieved",
        }

    # Format citations
    citation_text = ""
    for c in citations[:5]:
        if hasattr(c, "title"):
            citation_text += f"- {c.title}\n"
        elif isinstance(c, dict):
            citation_text += f"- {c.get('title', 'N/A')}\n"

    prompt = f"""You are evaluating whether retrieved medical articles are relevant to a question.

QUESTION: {question}

RETRIEVED ARTICLES:
{citation_text}

TASK: Are these articles relevant to answering the question?
- Relevant means the articles are about the same topic/condition/treatment
- Irrelevant means the articles are about something completely different

Respond with ONLY:
SCORE: 1 (if relevant) or 0 (if irrelevant)
REASON: One sentence explanation
"""

    try:
        response = _call_llm_judge(prompt)

        score = 1 if "SCORE: 1" in response or "SCORE:1" in response else 0
        reason_match = re.search(r"REASON:\s*(.+)", response, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else response[:200]

        return {
            "key": "relevance",
            "score": score,
            "reasoning": reason,
        }
    except Exception as e:
        return {
            "key": "relevance",
            "score": 0,
            "reasoning": f"Evaluation failed: {str(e)}",
        }


# ============================================================================
# All evaluators list for easy import
# ============================================================================

PROGRAMMATIC_EVALUATORS = [
    citation_accuracy,
    tool_trace_valid,
    has_citations,
    topic_coverage,
]

LLM_JUDGE_EVALUATORS = [
    faithfulness,
    answer_quality,
    relevance,
]

ALL_EVALUATORS = PROGRAMMATIC_EVALUATORS + LLM_JUDGE_EVALUATORS
