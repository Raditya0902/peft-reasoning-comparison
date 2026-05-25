"""Generate markdown report tables from evaluation results."""

import argparse
import json
from pathlib import Path


RUNS = ["base_zeroshot", "base_fewshot5", "lora", "dora", "ia3"]
ADAPTER_RUNS = ["lora", "dora", "ia3"]
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
    "base_zeroshot": "base_zeroshot_100.jsonl",
    "base_fewshot5": "base_fewshot5_100.jsonl",
    "lora": "lora_100.jsonl",
    "dora": "dora_100.jsonl",
    "ia3": "ia3_100.jsonl",
}


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_predictions(results_dir: Path) -> dict[str, list[dict]]:
    pred_dir = results_dir / "predictions"
    return {
        run: _load_jsonl(pred_dir / filename)
        for run, filename in JSONL_FILENAMES.items()
    }


def _load_baseline_summary(results_dir: Path) -> dict[str, dict]:
    summary_path = results_dir / "metrics" / "baseline_summary.json"
    with summary_path.open() as f:
        entries = json.load(f)
    return {entry["run"]: entry for entry in entries}


def _compute_main_metrics(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {
            "accuracy": 0.0,
            "extraction_failure_rate": 0.0,
            "avg_latency_ms": 0.0,
            "avg_output_tokens": 0.0,
        }
    n_correct = sum(1 for r in records if r["correct"])
    n_failures = sum(1 for r in records if r["extraction_failure"])
    return {
        "accuracy": n_correct / n,
        "extraction_failure_rate": n_failures / n,
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


def _build_main_results_table(run_records: dict[str, list[dict]]) -> str:
    metrics = {run: _compute_main_metrics(records) for run, records in run_records.items()}
    sorted_runs = sorted(RUNS, key=lambda r: metrics[r]["accuracy"], reverse=True)

    headers = ["Run", "Accuracy", "Extraction Failure Rate", "Avg Latency (ms)", "Avg Output Tokens"]
    rows = [
        [
            run,
            f"{metrics[run]['accuracy']:.4f}",
            f"{metrics[run]['extraction_failure_rate']:.4f}",
            f"{metrics[run]['avg_latency_ms']:.2f}",
            f"{metrics[run]['avg_output_tokens']:.2f}",
        ]
        for run in sorted_runs
    ]
    return "## Table 1 — Main Results\n\n" + _markdown_table(headers, rows)


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
    headers = ["Category"] + RUNS
    rows = [
        [category] + [_category_cell(run_records[run], category) for run in RUNS]
        for category in CATEGORIES
    ]
    return "## Table 2 — Per-Category Accuracy\n\n" + _markdown_table(headers, rows)


def _adapter_size_mb(run: str) -> str:
    path = Path(f"adapters/{run}/adapter_model.safetensors")
    if not path.exists():
        return "N/A"
    return f"{path.stat().st_size / (1024 ** 2):.2f}"


def _build_efficiency_table(summary_by_run: dict[str, dict]) -> str:
    headers = [
        "Run", "Adapter Size (MB)", "Trainable Params", "Trainable %",
        "Train Loss", "Eval Loss", "Avg Latency (ms)",
    ]
    rows = [
        [
            run,
            _adapter_size_mb(run),
            str(summary_by_run.get(run, {}).get("trainable_params", "N/A")),
            f"{summary_by_run.get(run, {}).get('trainable_pct', 0):.4f}",
            f"{summary_by_run.get(run, {}).get('final_train_loss', 0):.4f}",
            f"{summary_by_run.get(run, {}).get('final_eval_loss', 0):.4f}",
            f"{summary_by_run.get(run, {}).get('avg_latency_ms', 0):.2f}",
        ]
        for run in ADAPTER_RUNS
    ]
    return "## Table 3 — Efficiency\n\n" + _markdown_table(headers, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown report tables.")
    parser.add_argument("--results_dir", required=True, help="Path to results/ directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_records = _load_predictions(results_dir)
    summary_by_run = _load_baseline_summary(results_dir)

    table1 = _build_main_results_table(run_records)
    table2 = _build_per_category_table(run_records)
    table3 = _build_efficiency_table(summary_by_run)

    output = "\n\n".join([table1, table2, table3])
    print(output)

    out_path = results_dir / "report_tables.md"
    out_path.write_text(output + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
