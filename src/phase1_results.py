"""Generate Phase 1 tables from held-out predictions and training metadata."""

import argparse
import json
from itertools import combinations
from pathlib import Path

from src.phase1_parity import approved_for_phase1
from src.phase1_reference import check_zero_shot_prompts
from src.report_tables import (
    _build_main_results_table,
    _holm_adjust,
    _markdown_table,
    _mcnemar_exact,
    _rescore_runs,
)

ARMS = ("gold", "stripped", "self_generated")
REFERENCES = ("same_pipeline_lora_reference", "instruct_zeroshot_reference")
BINS = ((0, 99), (100, 149), (150, 199), (200, None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--accept_archive_drift", action="store_true")
    parser.add_argument("--parity_dir", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads((args.parity_dir / "parity.json").read_text())
    if not approved_for_phase1(gate, args.accept_archive_drift):
        raise ValueError("Both serial-equivalence checks must pass, with explicit approval for archive drift.")
    files = {arm: args.run_dir / f"{arm}_test.jsonl" for arm in ARMS}
    files.update(zip(REFERENCES, (
        args.reference_dir / "lora_reference.jsonl", args.parity_dir / "zeroshot_batched.jsonl",
    ), strict=True))
    common_fields = ("model", "model_revision", "dataset_fingerprint", "gpu_name", "model_dtype",
                     "batch_size", "seed", "split", "prompt_date", "few_shot", "versions")
    first = json.loads(files[REFERENCES[1]].with_suffix(".metadata.json").read_text())
    for name, path in files.items():
        observed = json.loads(path.with_suffix(".metadata.json").read_text())
        if any(observed[field] != first[field] for field in common_fields):
            raise ValueError(f"{name} used a different evaluation pipeline.")
    records, _ = _rescore_runs(Path("."), {name: p.resolve() for name, p in files.items()}, 1319)
    check_zero_shot_prompts(records[REFERENCES[1]], {name: records[name] for name in (*ARMS, REFERENCES[0])})
    metadata = {arm: json.loads((args.run_dir / f"{arm}_training.json").read_text()) for arm in ARMS}
    if not all(entry["status"] == "completed" and entry["target_format"] == arm for arm, entry in metadata.items()):
        raise ValueError("Training metadata must describe completed runs.")
    targets = {entry["targets_sha256"] for entry in metadata.values()}
    counts = {entry["target_length_tokens"]["count"] for entry in metadata.values()}
    configs = {json.dumps(entry["config"], sort_keys=True) for entry in metadata.values()}
    if len(counts) != 1 or len(configs) != 1:
        raise ValueError("All arms must use the same example count and training configuration.")
    if len(targets) != 1 or None in targets:
        raise ValueError("All arms must come from the same matched training archive.")

    bins = []
    for name, rows in records.items():
        for low, high in BINS:
            selected = [
                row for row in rows
                if row["num_output_tokens"] >= low and (high is None or row["num_output_tokens"] <= high)
            ]
            correct = sum(row["correct"] for row in selected)
            interval = f"{low}–{high}" if high is not None else f"{low}+"
            value = f"{correct / len(selected):.4f}" if selected else "—"
            bins.append([name, interval, str(len(selected)), str(correct), value])

    comparisons = []
    for left, right in [*combinations(ARMS, 2), (ARMS[0], REFERENCES[0])]:
        paired = [(a["correct"], b["correct"]) for a, b in zip(records[left], records[right], strict=True)]
        left_only = sum(a and not b for a, b in paired)
        right_only = sum(b and not a for a, b in paired)
        gap = (left_only - right_only) / len(paired)
        comparisons.append((left, right, left_only, right_only, gap, _mcnemar_exact(left_only, right_only)))
    format_rows = []
    for (left, right, a, b, gap, p), adjusted in zip(
        comparisons[:-1], _holm_adjust([row[-1] for row in comparisons[:-1]]), strict=True,
    ):
        decision = f"{left if gap > 0 else right} higher" if adjusted < 0.05 and abs(gap) > 0.02 else "not distinguishable"
        format_rows.append([left, right, str(a), str(b), f"{100 * gap:+.2f}", f"{p:.4g}", f"{adjusted:.4g}", decision])
    left, right, a, b, gap, p = comparisons[-1]

    length_rows = []
    for arm, entry in metadata.items():
        stats = entry["target_length_tokens"]
        length_rows.append([
            arm, str(stats["count"]), str(entry["candidate_count"]), str(stats["total_supervised_tokens"]),
            f"{stats['mean']:.2f}", *[str(stats["percentiles"][str(i)]) for i in (0, 10, 25, 50, 75, 90, 100)],
        ])
    sections = [
        "# Phase 1 Instruct ablation",
        "## Full test",
        _build_main_results_table(records),
        "## Matched-pool format comparisons",
        _markdown_table(["Left", "Right", "Left only", "Right only", "Gap (pp)", "Exact p", "Holm p", "Decision"], format_rows),
        "## Same-pipeline LoRA reference comparison",
        "The reference adapter was trained under the historical recipe and re-evaluated through the same pipeline. This contrast also changes the training pool and does not isolate the objective change.",
        _markdown_table(["Left", "Right", "Left only", "Right only", "Gap (pp)", "Exact p"], [[left, right, str(a), str(b), f"{100 * gap:+.2f}", f"{p:.4g}"]]),
        "## Accuracy by generated token count",
        "Length strata are descriptive because the intervention can change stratum membership.",
        _markdown_table(["Run", "Tokens", "N", "Correct", "Accuracy"], bins),
        "## Supervised target lengths",
        _markdown_table(["Run", "Accepted", "Candidates", "Supervised tokens", "Mean", "Min", "P10", "P25", "Median", "P75", "P90", "Max"], length_rows),
    ]
    with (args.run_dir / "report_tables.md").open("x") as f:
        f.write("\n\n".join(sections) + "\n")


if __name__ == "__main__":
    main()
