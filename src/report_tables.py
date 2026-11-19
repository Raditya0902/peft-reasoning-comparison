"""Generate markdown report tables from evaluation results."""

import argparse
import json
import random
from itertools import combinations
from math import comb
from pathlib import Path

from src.categorize import categorize
from src.extract_answer import extract_answer


RUNS = ["base_zeroshot", "base_fewshot5", "lora", "dora", "ia3", "ia3_convergence"]
ADAPTER_RUNS = ["lora", "dora", "ia3", "ia3_convergence"]
CATEGORIES = [
    "arithmetic",
    "fractions_percentages",
    "unit_conversion",
    "multi_hop",
    "algebraic",
    "comparison",
    "distractor_heavy",
    "uncategorized",
]
JSONL_FILENAMES = {
    "base_zeroshot": "base_zeroshot_full.jsonl",
    "base_fewshot5": "base_fewshot5_full.jsonl",
    "lora": "lora_full.jsonl",
    "dora": "dora_full.jsonl",
    "ia3": "ia3_full.jsonl",
    "ia3_convergence": "ia3_convergence_full.jsonl",
}


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_baseline_summary(results_dir: Path) -> dict[str, dict]:
    summary_path = results_dir / "metrics" / "baseline_summary.json"
    with summary_path.open() as f:
        entries = json.load(f)
    return {entry["run"]: entry for entry in entries}


def _is_answer_first(record: dict) -> bool:
    return record["model_output"].lstrip().startswith("####") or record["num_output_tokens"] <= 10


def _compute_main_metrics(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {
            "accuracy": 0.0,
            "extraction_failure_rate": 0.0,
            "answer_first_rate": 0.0,
            "avg_latency_ms": 0.0,
            "avg_output_tokens": 0.0,
        }
    n_correct = sum(1 for r in records if r["correct"])
    n_failures = sum(1 for r in records if r["extraction_failure"])
    return {
        "accuracy": n_correct / n,
        "extraction_failure_rate": n_failures / n,
        "answer_first_rate": sum(map(_is_answer_first, records)) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in records) / n,
        "avg_output_tokens": sum(r["num_output_tokens"] for r in records) / n,
    }


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return "\n".join([fmt_row(headers), separator] + [fmt_row(row) for row in rows])


def _bootstrap_ci(correct: list[bool], seed: int = 42) -> tuple[float, float]:
    """Percentile interval from 10,000 iid question resamples of the same size."""
    rng = random.Random(seed)
    n = len(correct)
    counts = sorted(sum(rng.choices(correct, k=n)) for _ in range(10_000))
    # Nearest-rank 2.5th and 97.5th percentiles.
    return counts[249] / n, counts[9749] / n


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    tail = sum(comb(n, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, (2 * tail) / (2 ** n))


def _rescore_runs(
    pred_dir: Path, filenames: dict[str, str], expected_n: int, *, expected_ids: set[str] | None = None
) -> tuple[dict[str, list[dict]], list[list[str]]]:
    scored = {}
    diagnostics = []
    if expected_ids is None:
        expected_ids = {f"gsm8k-{i}" for i in range(expected_n)}
    reference = None
    for run, filename in filenames.items():
        records = _load_jsonl(pred_dir / filename)
        if len(records) != expected_n or {r["id"] for r in records} != expected_ids:
            raise ValueError(f"{filename} must contain {expected_n} unique GSM8K IDs")
        records.sort(key=lambda r: int(r["id"].split("-")[-1]))
        examples = [(r["question"], r["gold_answer"]) for r in records]
        if reference is None:
            reference = examples
        if examples != reference or any(type(gold) is not int for _, gold in examples):
            raise ValueError(f"{filename} has mismatched questions or invalid gold answers")

        score_changes = tag_changes = calculator_outputs = 0
        for row in records:
            extraction = extract_answer(row["model_output"])
            correct = extraction.value == row["gold_answer"]
            tags = categorize(row["question"])
            score_changes += (
                extraction.value != row["predicted_answer"]
                or extraction.extraction_failure != row["extraction_failure"]
                or correct != row["correct"]
            )
            tag_changes += tags != row["category_tags"]
            calculator_outputs += "<<" in row["model_output"]
            row.update(
                correct=correct, extraction_failure=extraction.extraction_failure,
                predicted_answer=extraction.value, category_tags=tags,
            )
        scored[run] = records
        diagnostics.append([run, str(score_changes), str(tag_changes), str(calculator_outputs)])
    return scored, diagnostics


def _build_main_results_table(run_records: dict[str, list[dict]]) -> str:
    metrics = {run: _compute_main_metrics(records) for run, records in run_records.items()}
    sorted_runs = sorted(run_records, key=lambda r: metrics[r]["accuracy"], reverse=True)
    headers = ["Run", "Correct", "Accuracy", "Bootstrap 95% CI", "Avg Output Tokens",
               "Extraction Failure Rate", "Answer-first Rate"]
    rows = []
    for run in sorted_runs:
        records = run_records[run]
        correct = [r["correct"] for r in records]
        rows.append([
            run, f"{sum(correct)}/{len(correct)}",
            f"{metrics[run]['accuracy']:.4f}",
            "[{:.4f}, {:.4f}]".format(*_bootstrap_ci(correct)),
            f"{metrics[run]['avg_output_tokens']:.2f}",
            f"{metrics[run]['extraction_failure_rate']:.4f}",
            f"{metrics[run]['answer_first_rate']:.4f}",
        ])
    return _markdown_table(headers, rows)


def _category_cell(records: list[dict], category: str) -> str:
    if category == "uncategorized":
        subset = [r for r in records if not r["category_tags"]]
    else:
        subset = [r for r in records if category in r["category_tags"]]
    total = len(subset)
    if total == 0:
        return "—"
    correct = sum(1 for r in subset if r["correct"])
    return f"{correct}/{total} ({correct / total:.4f})"


def _build_per_category_table(run_records: dict[str, list[dict]]) -> str:
    reference = next(iter(run_records.values()))
    rows = [
        [category] + [_category_cell(records, category) for records in run_records.values()]
        for category in CATEGORIES
    ]
    n_tagged = sum(bool(r["category_tags"]) for r in reference)
    n_multi = sum(len(r["category_tags"]) > 1 for r in reference)
    n_assignments = sum(len(r["category_tags"]) for r in reference)
    note = (
        f"{n_tagged}/{len(reference)} questions match at least one lexical tag, "
        f"{n_multi} match multiple tags, and there are {n_assignments} tag assignments. "
        "Uncategorized questions match none of the rules."
    )
    return note + "\n\n" + _markdown_table(["Category", *run_records], rows)


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    for rank, idx in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running_max = max(running_max, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def _build_paired_table(run_records: dict[str, list[dict]]) -> str:
    rows, p_values = [], []
    for (left, a), (right, b) in combinations(run_records.items(), 2):
        paired = [(x["correct"], y["correct"]) for x, y in zip(a, b, strict=True)]
        left_only = sum(x and not y for x, y in paired)
        right_only = sum(y and not x for x, y in paired)
        p = _mcnemar_exact(left_only, right_only)
        p_values.append(p)
        rows.append([
            left, right, str(left_only), str(right_only),
            f"{100 * (left_only - right_only) / len(paired):+.2f}", f"{p:.4g}",
        ])
    for row, adjusted in zip(rows, _holm_adjust(p_values), strict=True):
        row.append(f"{adjusted:.4g}")
    return _markdown_table(
        ["Left", "Right", "Left Only Correct", "Right Only Correct", "Gap (pp)", "Exact p", "Holm p"],
        rows,
    )


def _build_efficiency_table(summary_by_run: dict[str, dict]) -> str:
    rows = [
        [run, f"{summary_by_run[run]['trainable_params']:,}", f"{summary_by_run[run]['trainable_pct']:.2f}%"]
        for run in ADAPTER_RUNS
    ]
    return _markdown_table(["Run", "Trainable Params", "Trainable Share (Reported)"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore archived GSM8K predictions and generate Markdown tables.")
    parser.add_argument("--results_dir", type=Path, required=True, help="Path to results/ directory")
    parser.add_argument("--output", type=Path, help="Output Markdown path")
    args = parser.parse_args()

    sections = [
        "# Phase 0 GSM8K Reanalysis",
        "Scores are recomputed from saved text with the repository extractor and stored gold integers. "
        "Two-sided exact McNemar tests are paired by question and corrected with Holm within each split. "
        "Bootstrap intervals describe question-level uncertainty conditional on these runs and seed. "
        "The historical pilot overlaps the full test set.",
    ]
    pilot_files = {run: f"{run}_100.jsonl" for run in RUNS if run != "ia3_convergence"}
    for title, filenames, n in [
        ("Full test", JSONL_FILENAMES, 1319),
        ("Historical pilot", pilot_files, 100),
    ]:
        records, diagnostics = _rescore_runs(args.results_dir / "predictions", filenames, n)
        sections.extend([
            f"## {title} ({n:,} questions)",
            "### Overall scores", _build_main_results_table(records),
            "### Category scores", _build_per_category_table(records),
            "### Paired comparisons", _build_paired_table(records),
            "### Archive checks",
            _markdown_table(["Run", "Changed Score Records", "Retagged Questions", "Outputs With `<<`"], diagnostics),
        ])
    sections.extend([
        "## Parameter efficiency",
        _build_efficiency_table(_load_baseline_summary(args.results_dir)),
    ])
    out_path = args.output or args.results_dir / "report_tables.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(sections) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
