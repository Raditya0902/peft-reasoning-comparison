"""Generate Phase 1 tables from held-out predictions and training metadata."""

import argparse
import json
import random
from itertools import combinations
from pathlib import Path

from src.phase1_parity import approved_for_phase1
from src.phase1_reference import check_zero_shot_prompts
from src.prompts import PHASE1_FROZEN_DATE
from src.report_tables import (
    _build_main_results_table,
    _compute_main_metrics,
    _holm_adjust,
    _markdown_table,
    _mcnemar_exact,
    _rescore_runs,
)

ARMS = ("gold", "stripped", "self_generated")
REFERENCES = ("same_pipeline_lora_reference", "instruct_zeroshot_reference")
BINS = ((0, 99), (100, 149), (150, 199), (200, None))
COMMON_EVAL_FIELDS = ("model", "model_revision", "dataset_fingerprint", "gpu_name", "model_dtype",
                      "batch_size", "seed", "split", "prompt_date", "few_shot", "versions")


def _read_metadata(predictions: Path) -> dict:
    return json.loads(predictions.with_suffix(".metadata.json").read_text())


def _probe_table(run_dir: Path, full_records: dict) -> str:
    variants = ("frozen", "alternate", "no_date")
    probe_rows = []
    shared_prompts = {}
    comparable = (*[field for field in COMMON_EVAL_FIELDS if field != "prompt_date"], "adapter_sha256")
    for arm in ARMS:
        files = {mode: run_dir / f"{arm}_probe_{mode}.jsonl" for mode in variants}
        full_metadata = _read_metadata(run_dir / f"{arm}_test.jsonl")
        full = {r["id"]: r for r in full_records[arm]}
        indices = random.Random(full_metadata["seed"]).sample(range(len(full)), 200)
        records, _ = _rescore_runs(
            Path("."), {key: path.resolve() for key, path in files.items()}, 200,
            expected_ids={f"gsm8k-{index}" for index in indices},
        )
        for mode, rows in records.items():
            metadata = _read_metadata(files[mode])
            if any(metadata[field] != full_metadata[field] for field in comparable):
                raise ValueError(f"Exploratory run {arm}/{mode} does not use the frozen inference pipeline.")
            if metadata["subset_size"] != 200 or metadata["omit_prompt_date"] != (mode == "no_date"):
                raise ValueError(f"Unexpected probe setup for {arm}/{mode}.")
            date = metadata["prompt_date"]
            expected_date = "18 Nov 2025" if mode == "alternate" else PHASE1_FROZEN_DATE
            if date != expected_date:
                raise ValueError(f"Unexpected date for {arm}/{mode}.")
            prompts = {}
            for row in rows:
                index = row["id"]
                key = f"Today Date: {PHASE1_FROZEN_DATE}\n"
                expected = full[index]["rendered_prompt"]
                if expected.count(key) != 1:
                    raise ValueError(f"The primary evaluation is missing its frozen date for {index}.")
                replacement = "" if mode == "no_date" else f"Today Date: {date}\n"
                if row["rendered_prompt"] != expected.replace(key, replacement, 1):
                    raise ValueError(f"Probe {arm}/{mode}/{index} differs outside its date line.")
                prompts[index] = (row["rendered_prompt"], row["prompt_token_ids"])
            if mode in shared_prompts and prompts != shared_prompts[mode]:
                raise ValueError(f"Probe prompts or token IDs differ across arms for {mode}.")
            shared_prompts[mode] = prompts
            metrics = _compute_main_metrics(rows)
            probe_rows.append([
                arm, "No date line" if mode == "no_date" else date, str(len(rows)),
                f"{metrics['accuracy']:.4f}", f"{metrics['avg_output_tokens']:.2f}",
                f"{metrics['answer_first_rate']:.4f}",
            ])
    return _markdown_table(["Arm", "Today Date", "N", "Accuracy", "Avg output tokens", "Answer-first rate"], probe_rows)


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
    first = _read_metadata(files[REFERENCES[1]])
    for name, path in files.items():
        observed = _read_metadata(path)
        if any(observed[field] != first[field] for field in COMMON_EVAL_FIELDS):
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
    probe = _probe_table(args.run_dir, records)
    validation_files = {arm: (args.run_dir / f"{arm}_validation.jsonl").resolve() for arm in ARMS}
    validation_n = _read_metadata(validation_files[ARMS[0]])["examples"]
    validation, _ = _rescore_runs(Path("."), validation_files, validation_n)
    sections = [
        "# Phase 1 Instruct ablation",
        "## Full test",
        _build_main_results_table(records),
        "## Matched-pool format comparisons",
        _markdown_table(["Left", "Right", "Left only", "Right only", "Gap (pp)", "Exact p", "Holm p", "Decision"], format_rows),
        "Gold vs stripped tests calculator annotation removal together with its change in supervised target length. Comparisons with the sampled teacher change the reasoning source, content, wording, and supervised target length. These are single-seed comparisons on the teacher-solvable subset.",
        "## Same-pipeline LoRA reference comparison",
        "The reference adapter was trained under the historical recipe and re-evaluated on the same prompts and pipeline. The training contrast jointly changes assistant masking, token normalization, training question selection, and system prompt alignment. Historical training runtime and precision provenance are incomplete.",
        _markdown_table(["Left", "Right", "Left only", "Right only", "Gap (pp)", "Exact p"], [[left, right, str(a), str(b), f"{100 * gap:+.2f}", f"{p:.4g}"]]),
        "## Validation decode",
        _build_main_results_table(validation),
        "## Exploratory prompt-date probe",
        "The same fixed 200 test questions are used in all nine evaluations. Only the Today Date line varies. This probe is outside the preregistered comparisons.",
        probe,
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
