"""Diagnose the saved parity check and report same-pipeline test references."""

import argparse
import json
from itertools import combinations
from pathlib import Path

from src.phase1_parity import compare
from src.report_tables import (
    _build_main_results_table,
    _holm_adjust,
    _load_jsonl,
    _markdown_table,
    _mcnemar_exact,
    _rescore_runs,
)

BASELINES = {"base_zeroshot": ("zeroshot", 0), "base_fewshot5": ("fewshot5", 5)}
ADAPTERS = ("lora", "dora", "ia3", "ia3_convergence")


def _pairs(left: list[dict], right: list[dict]) -> tuple[int, int, int, int, int]:
    a = {r["id"]: r for r in left}
    b = {r["id"]: r for r in right}
    if a.keys() != b.keys() or len(a) != len(left) or len(b) != len(right):
        raise ValueError("Paired samples must have the same unique IDs.")
    for key in a:
        if any(a[key][field] != b[key][field] for field in ("question", "gold_answer")):
            raise ValueError(f"Unmatched question or gold answer for {key}.")
    only_a = sum(a[k]["correct"] and not b[k]["correct"] for k in a)
    only_b = sum(b[k]["correct"] and not a[k]["correct"] for k in a)
    return len(a), sum(r["correct"] for r in left), sum(r["correct"] for r in right), only_a, only_b


def diagnose(parity_dir: Path, archive_dir: Path) -> str:
    saved_gate = json.loads((parity_dir / "parity.json").read_text())
    rows = []
    for historical, (name, shots) in BASELINES.items():
        archive = archive_dir / f"{historical}_full.jsonl"
        batched = parity_dir / f"{name}_batched.jsonl"
        serial = parity_dir / f"{name}_serial.jsonl"
        if compare(archive, batched, serial, shots) != saved_gate[name]:
            raise ValueError(f"Saved parity summary differs for {name}.")
        older, fast, replay = map(_load_jsonl, (archive, batched, serial))
        ids = {r["id"] for r in replay}
        old_replay = [r for r in older if r["id"] in ids]
        fast_replay = [r for r in fast if r["id"] in ids]
        for label, left, right in (
            ("Batched vs historical, full", fast, older),
            ("Batched vs historical, replay IDs", fast_replay, old_replay),
            ("Serial vs historical, replay IDs", replay, old_replay),
            ("Batched vs serial, replay IDs", fast_replay, replay),
        ):
            n, a, b, a_only, b_only = _pairs(left, right)
            rows.append([
                name, label, str(n), f"{a}/{n} ({100*a/n:.2f}%)", f"{b}/{n} ({100*b/n:.2f}%)",
                f"{100*(a-b)/n:+.2f}", str(a_only), str(b_only), f"{_mcnemar_exact(a_only,b_only):.4g}",
            ])
    headers = ["Baseline", "Comparison", "N", "Left accuracy", "Right accuracy",
               "Gap (pp)", "Left only correct", "Right only correct", "Exact p"]
    return "\n\n".join([
        "# V100 parity diagnosis",
        "Serial replay covers only the fixed sampled IDs. Historical and batched scores on those same IDs provide the comparable denominators. Tests are exact two-sided McNemar with unadjusted diagnostic p-values.",
        _markdown_table(headers, rows),
    ])


def check_zero_shot_prompts(reference: list[dict], candidates: dict[str, list[dict]]) -> None:
    expected = {r["id"]: (r["rendered_prompt"], r["prompt_token_ids"]) for r in reference}
    for name, rows in candidates.items():
        prompts = {r["id"]: (r["rendered_prompt"], r["prompt_token_ids"]) for r in rows}
        if prompts != expected:
            raise ValueError(f"{name} used different zero-shot prompts or input IDs.")


def _pair_stats(records: dict[str, list[dict]]) -> dict[tuple[str, str], dict]:
    stats = {}
    for a, b in combinations(records, 2):
        n, a_correct, b_correct, left, right = _pairs(records[a], records[b])
        stats[(a, b)] = {"gap": (a_correct-b_correct)/n, "left": left, "right": right, "p": _mcnemar_exact(left, right)}
    for row, adjusted in zip(stats.values(), _holm_adjust([s["p"] for s in stats.values()]), strict=True):
        row["adjusted"] = adjusted
        row["direction"] = (1 if row["gap"] > 0 else -1) if adjusted < 0.05 else 0
    return stats


def report(reference_dir: Path, parity_dir: Path, archive_dir: Path) -> str:
    files = {run: parity_dir / f"{name}_batched.jsonl" for run, (name, _) in BASELINES.items()}
    files.update({run: reference_dir / f"{run}_reference.jsonl" for run in ADAPTERS})
    expected = json.loads(files["base_zeroshot"].with_suffix(".metadata.json").read_text())
    keys = ("model", "model_revision", "dataset_fingerprint", "model_dtype", "gpu_name",
            "batch_size", "prompt_date", "seed", "split", "versions")
    for name, path in files.items():
        meta = json.loads(path.with_suffix(".metadata.json").read_text())
        if any(meta[k] != expected[k] for k in keys):
            raise ValueError(f"{name} uses a different evaluation pipeline.")
        if name in ADAPTERS and (not meta["adapter_sha256"] or meta["few_shot"] != 0):
            raise ValueError(f"{name} lacks adapter provenance or uses a different prompting condition.")

    records, _ = _rescore_runs(Path("."), {name: path.resolve() for name, path in files.items()}, 1319)
    check_zero_shot_prompts(records["base_zeroshot"], {name: records[name] for name in ADAPTERS})
    old, _ = _rescore_runs(archive_dir, {name: f"{name}_full.jsonl" for name in files}, 1319)
    new_pairs = _pair_stats(records)
    old_pairs = _pair_stats(old)

    def rank(runs):
        return sorted(runs, key=lambda run: sum(r["correct"] for r in runs[run]), reverse=True)

    def label(direction):
        return {1: "left higher", -1: "right higher", 0: "not distinguishable"}[direction]

    pairs = []
    for (a, b), row in new_pairs.items():
        pairs.append([a, b, str(row["left"]), str(row["right"]), f"{100*row['gap']:+.2f}",
                      f"{row['p']:.4g}", f"{row['adjusted']:.4g}", label(row["direction"]),
                      label(old_pairs[(a,b)]["direction"])])
    changed = sum(new_pairs[p]["direction"] != old_pairs[p]["direction"] for p in new_pairs)
    headlines = {
        "Zero-shot has higher accuracy than every other run": [(0, i, 1) for i in range(1, 6)],
        "2-epoch IA3 has higher accuracy than LoRA and DoRA": [(2, 4, -1), (3, 4, -1)],
        "2-epoch IA3 has higher accuracy than 10-epoch IA3": [(4, 5, 1)],
        "LoRA and DoRA are not distinguishable": [(2, 3, 0)],
        "Five-shot and 2-epoch IA3 are not distinguishable": [(1, 4, 0)],
    }
    names = list(files)
    claims = []
    for claim, tests in headlines.items():
        outcomes = [all(stats[(names[a], names[b])]["direction"] == d for a, b, d in tests)
                    for stats in (old_pairs, new_pairs)]
        claims.append([claim, *["yes" if v else "no" for v in outcomes], "yes" if outcomes[0] != outcomes[1] else "no"])
    sections = [
        "# Same-pipeline V100 test references",
        "Question-bootstrap percentile intervals use 10,000 resamples. Paired McNemar tests are two-sided and exact, with Holm adjustment across all 15 comparisons. All current comparisons use the FP16 V100 pipeline. The historical column is for drift diagnosis.",
        "Answer-first counts outputs that begin with `####` after leading whitespace or end within 10 generated tokens. Each output counts at most once. Current token counts include the first EOS when present.",
        _build_main_results_table(records),
        "## Paired tests",
        _markdown_table(["Left", "Right", "Left only", "Right only", "Gap (pp)", "Exact p", "Holm p", "Same pipeline", "Historical"], pairs),
        "## Changes from historical Phase 0",
        f"Historical point ranking: {' > '.join(rank(old))}.",
        f"Same-pipeline point ranking: {' > '.join(rank(records))}.",
        f"Pairwise decisions changed for {changed}/{len(new_pairs)} comparisons. Headline paired claims changed for {sum(row[-1]=='yes' for row in claims)}/{len(claims)} claims.",
        _markdown_table(["Claim", "Historical", "Same pipeline", "Changed"], claims),
    ]
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity_dir", type=Path, required=True)
    parser.add_argument("--archive_dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--reference_dir", type=Path, help="Generate reference report when supplied.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (report(args.reference_dir, args.parity_dir, args.archive_dir)
              if args.reference_dir else diagnose(args.parity_dir, args.archive_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(result + "\n")


if __name__ == "__main__":
    main()
