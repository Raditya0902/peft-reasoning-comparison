"""Compare full batched scores to the archive and raw tokens to a serial replay."""

import argparse
import json
import random
from pathlib import Path

from src.report_tables import _load_jsonl

FULL_SIZE = 1319
REPLAY_SIZE = 128
SEED = 42
MAX_ACCURACY_GAP = 0.01
MIN_EXACT_TOKEN_MATCH = 0.95


def _index(path: Path) -> dict[str, dict]:
    rows = _load_jsonl(path)
    indexed = {row["id"]: row for row in rows}
    if len(rows) != len(indexed):
        raise ValueError(f"Duplicate example IDs in {path}")
    return indexed


def compare(archive: Path, batched: Path, serial: Path, few_shot: int) -> dict:
    old, fast, replay = map(_index, (archive, batched, serial))
    expected = {f"gsm8k-{i}" for i in range(FULL_SIZE)}
    replay_ids = {f"gsm8k-{i}" for i in random.Random(SEED).sample(range(FULL_SIZE), REPLAY_SIZE)}
    if old.keys() != expected or fast.keys() != expected or replay.keys() != replay_ids:
        raise ValueError("Expected both full test artifacts and the preregistered serial replay IDs.")

    fast_meta, serial_meta = (
        json.loads(path.with_suffix(".metadata.json").read_text()) for path in (batched, serial)
    )
    for key in ("model", "model_revision", "dataset_fingerprint", "device", "gpu_name", "model_dtype", "versions", "prompt_date"):
        if fast_meta[key] != serial_meta[key]:
            raise ValueError(f"Serial and batched settings differ for {key}.")
    for meta in (fast_meta, serial_meta):
        if "V100" not in (meta["gpu_name"] or ""):
            raise ValueError("Parity requires the budgeted V100.")
        if (meta["few_shot"], meta["seed"], meta["split"], meta["adapter"]) != (few_shot, SEED, "test", None):
            raise ValueError("Unexpected baseline inference settings.")
    if serial_meta["batch_size"] != 1 or fast_meta["batch_size"] < 2:
        raise ValueError("Expected batch size one for serial replay and a larger fast batch size.")

    for comparison in (old, replay):
        for key, row in comparison.items():
            if any(row[field] != fast[key][field] for field in ("question", "gold_answer")):
                raise ValueError(f"Question or answer mismatch for {key}.")

    for key, row in replay.items():
        if any(row[field] != fast[key][field] for field in ("rendered_prompt", "prompt_token_ids")):
            raise ValueError(f"Serial and batched prompts differ for {key}.")

    old_correct = sum(row["correct"] for row in old.values())
    fast_correct = sum(row["correct"] for row in fast.values())
    exact_tokens = sum(row["output_token_ids"] == fast[key]["output_token_ids"] for key, row in replay.items())
    agreement = exact_tokens / len(replay)
    gap = (fast_correct - old_correct) / len(old)
    return {
        "archived_accuracy": old_correct / len(old),
        "batched_accuracy": fast_correct / len(fast),
        "accuracy_delta": gap,
        "replay_exact_token_sequences": exact_tokens,
        "replay_size": len(replay),
        "replay_exact_token_sequence_rate": agreement,
        "archive_exact_decoded_text_count": sum(row["model_output"] == fast[key]["model_output"] for key, row in old.items()),
        "full_size": len(fast),
        "batched_wall_seconds": fast_meta["wall_seconds"],
        "serial_wall_seconds": serial_meta["wall_seconds"],
        "gpu_name": fast_meta["gpu_name"],
        "prompt_date": fast_meta["prompt_date"],
        "model_dtype": fast_meta["model_dtype"],
        "model_revision": fast_meta["model_revision"],
        "batch_size": fast_meta["batch_size"],
        "passed": abs(gap) <= MAX_ACCURACY_GAP and agreement >= MIN_EXACT_TOKEN_MATCH,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--archive_dir", type=Path, default=Path("results/predictions"))
    args = parser.parse_args()
    results = {
        name: compare(
            args.archive_dir / f"base_{name}_full.jsonl",
            args.run_dir / f"{name}_batched.jsonl",
            args.run_dir / f"{name}_serial.jsonl",
            few_shot,
        )
        for name, few_shot in (("zeroshot", 0), ("fewshot5", 5))
    }
    output = args.run_dir / "parity.json"
    with output.open("x") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(json.dumps(results, indent=2))
    if not all(run["passed"] for run in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
