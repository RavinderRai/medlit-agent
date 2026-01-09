"""Input validation utilities."""

import re

import structlog

logger = structlog.get_logger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None):
        self.message = message
        self.field = field
        super().__init__(message)


# Patterns for validation
MIN_QUESTION_LENGTH = 10
MAX_QUESTION_LENGTH = 1000
DANGEROUS_PATTERNS = [
    r"<script",
    r"javascript:",
    r"on\w+\s*=",
    r"data:text/html",
]


def validate_question(question: str) -> str:
    """Validate and clean a medical question.

    Args:
        question: The user's question

    Returns:
        Cleaned question string

    Raises:
        ValidationError: If the question is invalid
    """
    if not question:
        raise ValidationError("Question cannot be empty", field="question")

    # Strip whitespace
    question = question.strip()

    # Check length
    if len(question) < MIN_QUESTION_LENGTH:
        raise ValidationError(
            f"Question too short (minimum {MIN_QUESTION_LENGTH} characters)",
            field="question",
        )

    if len(question) > MAX_QUESTION_LENGTH:
        raise ValidationError(
            f"Question too long (maximum {MAX_QUESTION_LENGTH} characters)",
            field="question",
        )

    # Check for dangerous patterns
    question_lower = question.lower()
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, question_lower, re.IGNORECASE):
            logger.warning("Dangerous pattern detected in question", pattern=pattern)
            raise ValidationError(
                "Question contains invalid content",
                field="question",
            )

    return question


def sanitize_input(text: str, max_length: int | None = None) -> str:
    """Sanitize text input.

    Args:
        text: Input text to sanitize
        max_length: Optional maximum length

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Strip whitespace
    text = text.strip()

    # Remove null bytes
    text = text.replace("\x00", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length]
        logger.debug("Input truncated", max_length=max_length)

    return text

def extract_entities(question: str) -> dict[str, list[str]]:
    """Extract potential medical entities from a question.

    This is a simple pattern-based extraction, not NER.

    Args:
        question: The question to analyze

    Returns:
        Dictionary with entity types and found entities
    """
    entities: dict[str, list[str]] = {
        "drugs": [],
        "conditions": [],
        "populations": [],
    }

    # Common drug patterns (simplified)
    drug_patterns = [
        r"\b(aspirin|ibuprofen|metformin|lisinopril|atorvastatin)\b",
        r"\b\w+(?:mab|nib|vir|pril|sartan|statin|olol|azole)\b",
    ]

    # Common condition patterns
    condition_patterns = [
        r"\b(diabetes|cancer|hypertension|asthma|depression|arthritis)\b",
        r"\b\w+(?:itis|osis|emia|opathy|oma)\b",
    ]

    # Population patterns
    population_patterns = [
        r"\b(adult|child|elderly|pregnant|women|men|patient)\w*\b",
    ]

    question_lower = question.lower()

    for pattern in drug_patterns:
        matches = re.findall(pattern, question_lower, re.IGNORECASE)
        entities["drugs"].extend(matches)

    for pattern in condition_patterns:
        matches = re.findall(pattern, question_lower, re.IGNORECASE)
        entities["conditions"].extend(matches)

    for pattern in population_patterns:
        matches = re.findall(pattern, question_lower, re.IGNORECASE)
        entities["populations"].extend(matches)

    # Deduplicate
    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities
