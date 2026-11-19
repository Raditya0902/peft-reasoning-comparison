"""Pair the same-pipeline IA3 predictions with their historical counterparts."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from src.phase1_reference import _pairs
from src.report_tables import _load_jsonl, _markdown_table, _mcnemar_exact

QUESTION_COUNT = 1319
CAP = 512
RUNS = {"ia3": "checkpoint-750", "ia3_convergence": "checkpoint-3750"}


def _digest(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def _lengths(counts: list[int]) -> dict:
    ordered = sorted(counts)
    positions = (0, 5, 25, 50, 75, 95, 100)
    return {
        "mean": mean(counts),
        **{f"p{p}": ordered[round((len(counts) - 1)*p/100)] for p in positions},
    }


def _summarize(rows: list[dict], eos: set[int], *, historical: bool) -> dict:
    lengths = [r["num_output_tokens"] for r in rows]
    if historical:
        truncated = None
    else:
        truncated = 0
        for row in rows:
            ids = row["output_token_ids"]
            if len(ids) != row["num_output_tokens"] or not ids or len(ids) > CAP:
                raise ValueError(f"Unexpected token length for {row['id']}.")
            if ids[-1] not in eos:
                if len(ids) != CAP:
                    raise ValueError(f"Unexpected non-EOS stop before cap for {row['id']}.")
                truncated += 1
    return {
        "accuracy": sum(r["correct"] for r in rows)/len(rows),
        "extraction_failures": sum(r["extraction_failure"] for r in rows),
        "length": _lengths(lengths),
        "cap_hits": sum(n >= CAP for n in lengths),
        "confirmed_truncated": truncated,
        "calculator_marker_count": sum("<<" in r["model_output"] for r in rows),
    }


def _row(row: dict) -> dict:
    return {field: row[field] for field in (
        "predicted_answer", "correct", "num_output_tokens", "extraction_failure",
    )}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--archive_dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--adapter_dir", type=Path, default=Path("adapters"))
    parser.add_argument("--generation_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    eos_arg = json.loads(args.generation_config.read_text())["eos_token_id"]
    eos = {eos_arg} if isinstance(eos_arg, int) else set(eos_arg)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stats, lengths, discordant, digests, strata = {}, [], [], [], []
    for name, checkpoint in RUNS.items():
        old = _load_jsonl(args.archive_dir / f"{name}_full.jsonl")
        fast = _load_jsonl(args.reference_dir / f"{name}_reference.jsonl")
        n, b, a, b_only, a_only = _pairs(fast, old)
        if n != QUESTION_COUNT:
            raise ValueError(f"Expected {QUESTION_COUNT} rows for {name}.")
        a_by_id = {r["id"]: r for r in old}
        pairs = [(r, a_by_id[r["id"]]) for r in fast]
        high = [pair for pair in pairs if any(r["num_output_tokens"] >= CAP for r in pair)]
        extraction = [pair for pair in pairs if any(r["extraction_failure"] for r in pair)]

        def gain(group):
            return sum(int(f["correct"])-int(h["correct"]) for f, h in group)

        short_archive = [pair for pair in pairs if pair[1]["num_output_tokens"] <= 10]
        other_archive = [pair for pair in pairs if pair[1]["num_output_tokens"] > 10]
        clean = [pair for pair in pairs if all(
            r["num_output_tokens"] < CAP and not r["extraction_failure"] for r in pair
        )]
        adapter = args.adapter_dir / name
        weights, config_path = adapter / "adapter_model.safetensors", adapter / "adapter_config.json"
        got = json.loads((args.reference_dir / f"{name}_reference.metadata.json").read_text())
        current = _digest(weights)
        checkpoint_digest = _digest(adapter / checkpoint / weights.name)
        config = json.loads(config_path.read_text())
        checkpoint_config_equal = config == json.loads((adapter / checkpoint / config_path.name).read_text())
        if current != got["adapter_sha256"]:
            raise ValueError(f"The local {name} weights differ from the rerun metadata.")
        arch_paths = sorted({r["adapter_path"] for r in old})
        a_stats = _summarize(old, eos, historical=True)
        b_stats = _summarize(fast, eos, historical=False)
        stats[name] = {
            "n": n, "rerun": b_stats, "historical": a_stats,
            "both_correct": b-b_only, "rerun_only_correct": b_only, "historical_only_correct": a_only,
            "both_incorrect": n-b-a_only, "exact_mcnemar_p": _mcnemar_exact(b_only, a_only),
            "predicted_answer_disagreements": sum(f["predicted_answer"] != h["predicted_answer"] for f, h in pairs),
            "historical_10_tokens_or_less": {
                "questions": len(short_archive), "net_rerun_correct": gain(short_archive),
                "historical_start_with_answer_marker": sum(h["model_output"].lstrip().startswith("####") for _, h in short_archive),
                "historical_correct": sum(h["correct"] for _, h in short_archive),
                "rerun_correct": sum(f["correct"] for f, _ in short_archive),
                "rerun_still_10_tokens_or_less": sum(f["num_output_tokens"] <= 10 for f, _ in short_archive),
            },
            "historical_over_10_tokens": {"questions": len(other_archive), "net_rerun_correct": gain(other_archive)},
            "cap_in_either_run": {"questions": len(high), "net_rerun_correct": gain(high)},
            "extraction_failure_in_either_run": {"questions": len(extraction), "net_rerun_correct": gain(extraction)},
            "neither_cap_nor_failure": {
                "questions": len(clean),
                "net_rerun_correct": gain(clean),
            },
            "provenance": {
                "archived_adapter_paths": arch_paths,
                "rerun_adapter_sha256": got["adapter_sha256"],
                "local_adapter_sha256": current,
                "local_config_sha256": _digest(config_path),
                "local_checkpoint": checkpoint,
                "local_checkpoint_sha256": checkpoint_digest,
                "local_checkpoint_config_equal": checkpoint_config_equal,
                "feedforward_modules": config.get("feedforward_modules"),
                "target_modules": config.get("target_modules"),
            },
        }
        for label, stat in (("Same pipeline", b_stats), ("Historical", a_stats)):
            dist = stat["length"]
            truncated = stat["confirmed_truncated"]
            lengths.append([
                name, label, f"{100*stat['accuracy']:.2f}%", f"{dist['mean']:.2f}",
                *[str(dist[f"p{p}"]) for p in (0, 5, 25, 50, 75, 95, 100)],
                f"{stat['extraction_failures']}/{n} ({100*stat['extraction_failures']/n:.2f}%)",
                f"{stat['cap_hits']}/{n} ({100*stat['cap_hits']/n:.2f}%)",
                "unknown" if truncated is None else str(truncated),
            ])
        discordant.append([
            name, str(b-b_only), str(b_only), str(a_only), str(n-b-a_only),
            str(stats[name]["predicted_answer_disagreements"]), f"{_mcnemar_exact(b_only,a_only):.4g}",
        ])
        digests.append([name, ", ".join(arch_paths), checkpoint, current,
                        "yes" if current == checkpoint_digest and checkpoint_config_equal else "no"])
        for scope in ("cap_in_either_run", "extraction_failure_in_either_run", "neither_cap_nor_failure",
                      "historical_10_tokens_or_less", "historical_over_10_tokens"):
            data = stats[name][scope]
            strata.append([name, scope.replace("_", " "), str(data["questions"]), f"{data['net_rerun_correct']:+}"])
        with (args.output_dir / f"{name}_per_question.jsonl").open("x") as f:
            for fresh, prior in pairs:
                f.write(json.dumps({
                    "id": fresh["id"],
                    "gold_answer": fresh["gold_answer"],
                    "same_pipeline": {**_row(fresh), "truncated": fresh["output_token_ids"][-1] not in eos},
                    "historical": {**_row(prior), "cap_hit": prior["num_output_tokens"] >= CAP},
                }) + "\n")

    (args.output_dir / "diagnosis.json").write_text(
        json.dumps({"stop_token_ids": sorted(eos), "generation_cap": CAP, "runs": stats}, indent=2) + "\n"
    )
    sections = [
        "# IA3 archive-drift diagnosis",
        "## Lengths, extraction, and generation stops",
        "Recorded lengths include EOS when present. Historical cap hits are upper bounds on truncation because the historical JSONL omits raw output IDs. In the rerun, a capped output without EOS is confirmed truncated.",
        _markdown_table(["Run", "Source", "Accuracy", "Mean", "Min", "P5", "P25", "Median",
                         "P75", "P95", "Max", "Extraction failures", "Cap hits", "Confirmed truncation"], lengths),
        "## Paired outcomes on identical questions",
        _markdown_table(["Run", "Both correct", "Rerun only", "Historical only", "Both wrong", "Different extracted answer", "Exact p"], discordant),
        _markdown_table(["Run", "Subset", "Questions", "Net rerun correct"], strata),
        "The last two rows for each run stratify by historical output length. They describe where the gain occurred. The historical length belongs to the generated response and does not identify a causal intervention.",
        "## Adapter identity",
        "The final local weights match the V100 rerun hash. The historical JSONLs record directories and omit checkpoint hashes. The local final checkpoint match establishes the rerun epoch, while it cannot attest to the historical file contents.",
        _markdown_table(["Run", "Historical adapter path", "Local final checkpoint", "Rerun SHA-256", "Local checkpoint weights and config match"], digests),
    ]
    (args.output_dir / "diagnosis.md").write_text("\n\n".join(sections) + "\n")


if __name__ == "__main__":
    main()
