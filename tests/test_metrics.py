"""Tests for metric computation utilities."""

import pytest
from src.report_tables import _category_cell, _compute_main_metrics


def _rec(
    correct: bool,
    extraction_failure: bool = False,
    category_tags: list[str] | None = None,
    latency_ms: float = 100.0,
    num_output_tokens: int = 50,
) -> dict:
    return {
        "correct": correct,
        "extraction_failure": extraction_failure,
        "category_tags": category_tags if category_tags is not None else [],
        "latency_ms": latency_ms,
        "num_output_tokens": num_output_tokens,
    }


@pytest.mark.unit
def test_exact_match_correct():
    metrics = _compute_main_metrics([_rec(correct=True)])
    assert metrics["accuracy"] == 1.0


@pytest.mark.unit
def test_exact_match_incorrect():
    metrics = _compute_main_metrics([_rec(correct=False)])
    assert metrics["accuracy"] == 0.0


@pytest.mark.unit
def test_category_accuracy():
    records = [
        _rec(correct=True, category_tags=["arithmetic"]),
        _rec(correct=True, category_tags=["arithmetic"]),
        _rec(correct=False, category_tags=["arithmetic"]),
    ]
    cell = _category_cell(records, "arithmetic")
    assert cell == f"2/3 ({2 / 3:.4f})"


@pytest.mark.unit
def test_extraction_failure_rate():
    records = [
        _rec(correct=True, extraction_failure=False),
        _rec(correct=True, extraction_failure=False),
        _rec(correct=True, extraction_failure=False),
        _rec(correct=True, extraction_failure=False),
        _rec(correct=False, extraction_failure=True),
    ]
    metrics = _compute_main_metrics(records)
    assert metrics["extraction_failure_rate"] == pytest.approx(0.20)


@pytest.mark.unit
def test_empty_category():
    records = [
        _rec(correct=True, category_tags=[]),
        _rec(correct=False, category_tags=["arithmetic"]),
    ]
    cell = _category_cell(records, "uncategorized")
    assert cell == f"1/1 ({1 / 1:.4f})"
