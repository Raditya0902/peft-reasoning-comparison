"""Tests for answer extraction from model outputs."""

import pytest
from src.extract_answer import extract_answer, ExtractionResult


@pytest.mark.unit
def test_gsm8k_boxed_format():
    result = extract_answer("#### 42")
    assert result.value == 42
    assert not result.extraction_failure


@pytest.mark.unit
def test_natural_language_answer_is():
    result = extract_answer("The answer is 42.")
    assert result.value == 42
    assert not result.extraction_failure


@pytest.mark.unit
def test_final_answer_with_currency_and_commas():
    result = extract_answer("Final answer: $1,200")
    assert result.value == 1200
    assert not result.extraction_failure


@pytest.mark.unit
def test_therefore_negative_number():
    result = extract_answer("Therefore: -5")
    assert result.value == -5
    assert not result.extraction_failure


@pytest.mark.unit
def test_no_answer_present():
    result = extract_answer("The sky is blue and clouds are white.")
    assert result.extraction_failure
    assert result.value is None
