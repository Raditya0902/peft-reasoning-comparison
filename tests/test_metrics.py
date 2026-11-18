"""Tests for metric computation utilities."""

import json

import pytest
from src.report_tables import (
    _bootstrap_ci,
    _category_cell,
    _compute_main_metrics,
    _holm_adjust,
    _mcnemar_exact,
    _rescore_runs,
)


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


@pytest.mark.unit
def test_bootstrap_uses_empirical_accuracy_distribution():
    assert _bootstrap_ci([False] * 12) == (0.0, 0.0)
    assert _bootstrap_ci([True] * 12) == (1.0, 1.0)
    assert _bootstrap_ci([True, False] * 6) == (3 / 12, 9 / 12)


@pytest.mark.unit
def test_exact_mcnemar_has_two_tails_and_is_symmetric():
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(0, 6) == _mcnemar_exact(6, 0) == 2 / 64
    assert _mcnemar_exact(1, 5) == 14 / 64


@pytest.mark.unit
def test_holm_preserves_pair_order_and_monotonicity():
    assert _holm_adjust([0.04, 0.01, 0.025]) == pytest.approx([0.05, 0.03, 0.05])


@pytest.mark.unit
def test_rescoring_uses_text_and_rejects_mismatched_pairs(tmp_path):
    row = {
        **_rec(False),
        "id": "gsm8k-0",
        "question": "What is the total?",
        "gold_answer": 4,
        "predicted_answer": 3,
        "model_output": "#### 4",
    }
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    left.write_text(json.dumps(row))
    right.write_text(json.dumps({**row, "question": "What remains?"}))
    runs, diagnostics = _rescore_runs(tmp_path, {"left": left.name}, expected_n=1)
    assert runs["left"][0]["correct"] is True
    assert runs["left"][0]["category_tags"] == ["arithmetic"]
    assert diagnostics == [["left", "1", "1", "0"]]
    assert json.loads(left.read_text()) == row
    with pytest.raises(ValueError, match="mismatched questions"):
        _rescore_runs(tmp_path, {"left": left.name, "right": right.name}, expected_n=1)
