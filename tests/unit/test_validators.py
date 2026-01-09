"""Tests for input validators."""

import pytest

from medlit.utils.validators import (
    validate_question,
    sanitize_input,
    extract_entities,
    ValidationError,
)


class TestValidateQuestion:
    """Tests for question validation."""

    def test_valid_question(self):
        """Test valid question passes."""
        question = "What are the side effects of aspirin?"
        result = validate_question(question)
        assert result == question.strip()

    def test_empty_question(self):
        """Test empty question raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_question("")
        assert exc_info.value.field == "question"

    def test_short_question(self):
        """Test too short question raises error."""
        with pytest.raises(ValidationError):
            validate_question("Hi")

    def test_long_question(self):
        """Test too long question raises error."""
        with pytest.raises(ValidationError):
            validate_question("x" * 1001)

    def test_dangerous_pattern(self):
        """Test dangerous pattern is rejected."""
        with pytest.raises(ValidationError):
            validate_question("What is <script>alert(1)</script>?")

    def test_strips_whitespace(self):
        """Test whitespace is stripped."""
        result = validate_question("  What is aspirin?  ")
        assert result == "What is aspirin?"


class TestSanitizeInput:
    """Tests for input sanitization."""

    def test_empty_input(self):
        """Test empty input."""
        assert sanitize_input("") == ""

    def test_strips_whitespace(self):
        """Test whitespace stripping."""
        assert sanitize_input("  hello  ") == "hello"

    def test_removes_null_bytes(self):
        """Test null byte removal."""
        assert sanitize_input("hello\x00world") == "helloworld"

    def test_normalizes_whitespace(self):
        """Test whitespace normalization."""
        assert sanitize_input("hello    world") == "hello world"

    def test_truncates(self):
        """Test truncation."""
        result = sanitize_input("hello world", max_length=5)
        assert result == "hello"

class TestExtractEntities:
    """Tests for entity extraction."""

    def test_extract_drug(self):
        """Test drug extraction."""
        entities = extract_entities("Is aspirin good for prevention?")
        assert "aspirin" in entities["drugs"]

    def test_extract_condition(self):
        """Test condition extraction."""
        entities = extract_entities("How is diabetes treated?")
        assert "diabetes" in entities["conditions"]

    def test_extract_population(self):
        """Test population extraction."""
        entities = extract_entities("Is it safe for elderly patients?")
        assert any("elderly" in p for p in entities["populations"])
