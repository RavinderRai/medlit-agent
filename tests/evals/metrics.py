"""Custom evaluation metrics for MedLit Agent."""

from typing import Optional

from medlit.models import AgentResponse, EvidenceQuality


def citation_quality_score(response: AgentResponse) -> float:
    """Calculate citation quality score.

    Evaluates:
    - Number of citations
    - Diversity of sources (journals)
    - Recency of citations
    - Quality of study types

    Args:
        response: Agent response to evaluate

    Returns:
        Score from 0.0 to 1.0
    """
    if not response.citations:
        return 0.0

    citations = response.citations
    scores = []

    # Number of citations (target: 3-8)
    count = len(citations)
    if count == 0:
        count_score = 0.0
    elif count < 3:
        count_score = count / 3
    elif count <= 8:
        count_score = 1.0
    else:
        count_score = 0.9  # Slight penalty for too many

    scores.append(count_score)

    # Journal diversity
    journals = set(c.journal for c in citations if c.journal)
    if citations:
        diversity_score = min(1.0, len(journals) / len(citations))
    else:
        diversity_score = 0.0
    scores.append(diversity_score)

    # Recency (last 5 years gets full score)
    from datetime import date

    current_year = date.today().year
    recent_count = sum(1 for c in citations if c.year and c.year >= current_year - 5)
    if citations:
        recency_score = recent_count / len(citations)
    else:
        recency_score = 0.0
    scores.append(recency_score)

    return sum(scores) / len(scores)


def evidence_quality_score(response: AgentResponse) -> float:
    """Calculate evidence quality score.

    Based on the types of studies cited and overall evidence assessment.

    Args:
        response: Agent response to evaluate

    Returns:
        Score from 0.0 to 1.0
    """
    if not response.evidence:
        return 0.0

    quality_scores = {
        EvidenceQuality.HIGH: 1.0,
        EvidenceQuality.MODERATE: 0.66,
        EvidenceQuality.LOW: 0.33,
        EvidenceQuality.UNKNOWN: 0.0,
    }

    return quality_scores.get(response.evidence.quality, 0.0)


def answer_completeness_score(
    response: AgentResponse,
    expected_topics: Optional[list[str]] = None,
) -> float:
    """Calculate answer completeness score.

    Args:
        response: Agent response to evaluate
        expected_topics: Optional list of topics that should be covered

    Returns:
        Score from 0.0 to 1.0
    """
    if not response.answer:
        return 0.0

    scores = []

    # Basic length check (too short is bad)
    word_count = len(response.answer.split())
    if word_count < 50:
        length_score = word_count / 50
    elif word_count <= 500:
        length_score = 1.0
    else:
        length_score = 0.9  # Slight penalty for very long

    scores.append(length_score)

    # Topic coverage if expected topics provided
    if expected_topics:
        answer_lower = response.answer.lower()
        covered = sum(
            1 for topic in expected_topics
            if any(word in answer_lower for word in topic.lower().split())
        )
        topic_score = covered / len(expected_topics)
        scores.append(topic_score)

    # Check for key components
    has_disclaimer = "disclaimer" in response.answer.lower() or response.disclaimer
    has_limitations = response.evidence and response.evidence.limitations
    has_citations_in_text = "[PMID" in response.answer or "PMID:" in response.answer

    component_score = (
        (0.3 if has_disclaimer else 0) +
        (0.3 if has_limitations else 0) +
        (0.4 if has_citations_in_text else 0)
    )
    scores.append(component_score)

    return sum(scores) / len(scores)


def overall_response_score(
    response: AgentResponse,
    expected_topics: Optional[list[str]] = None,
) -> dict[str, float]:
    """Calculate overall response quality score.

    Args:
        response: Agent response to evaluate
        expected_topics: Optional list of expected topics

    Returns:
        Dictionary with individual and overall scores
    """
    citation_score = citation_quality_score(response)
    evidence_score = evidence_quality_score(response)
    completeness_score = answer_completeness_score(response, expected_topics)

    # Weighted average
    overall = (
        citation_score * 0.3 +
        evidence_score * 0.3 +
        completeness_score * 0.4
    )

    return {
        "citation_quality": citation_score,
        "evidence_quality": evidence_score,
        "completeness": completeness_score,
        "overall": overall,
    }
