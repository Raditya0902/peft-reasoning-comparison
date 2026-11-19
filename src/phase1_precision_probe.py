"""Measure first-token precision sensitivity on the historical short-answer cohort."""

import argparse
import gc
import json
from pathlib import Path

import torch

from src.evaluate import _generate_batch, _load_eval_adapter, _load_model_and_tokenizer
from src.report_tables import _load_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--zero_shot", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    historical = {r["id"]: r for r in _load_jsonl(args.historical) if r["num_output_tokens"] <= 10}
    rows = [r for r in _load_jsonl(args.zero_shot) if r["id"] in historical]
    if len(rows) != len(historical) or not all(
        r["question"] == historical[r["id"]]["question"] for r in rows
    ):
        raise ValueError("Historical and frozen prompts contain different question sets.")
    outcomes, checksums = {}, {}
    batch_size = 8
    for dtype in ("float16", "bfloat16"):
        print(f"First-token precision probe: {dtype}", flush=True)
        model, tokenizer = _load_model_and_tokenizer(args.model_path, dtype, device="cpu")
        model, checksums[dtype] = _load_eval_adapter(model, str(args.adapter))
        current = []
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset+batch_size]
            generated = _generate_batch(model, tokenizer, [r["rendered_prompt"] for r in batch], max_new_tokens=1)
            for row, result in zip(batch, generated, strict=True):
                if row["prompt_token_ids"] != result.prompt_token_ids:
                    raise ValueError(f"Frozen input IDs differ at {row['id']}.")
                current.append({
                    "id": row["id"], "first_output_token_ids": result.output_token_ids,
                    "starts_with_answer_marker": result.text.lstrip().startswith("####"),
                })
        outcomes[dtype] = current
        del model, tokenizer
        gc.collect()
    if len(set(checksums.values())) != 1:
        raise ValueError("The two runs loaded different adapter bytes.")

    pairs = list(zip(outcomes["float16"], outcomes["bfloat16"], strict=True))
    p = {
        "scope": "Historical 10-epoch outputs at most 10 tokens. Frozen prompt, actual final IA3 adapter, CPU, and one decoded token.",
        "questions": len(rows),
        "adapter_sha256": checksums["float16"],
        "first_token_differences": sum(a["first_output_token_ids"] != b["first_output_token_ids"] for a, b in pairs),
        "answer_marker_first_token_counts": {
            dtype: sum(row["starts_with_answer_marker"] for row in outcomes[dtype]) for dtype in outcomes
        },
        "outcomes": outcomes,
    }
    with args.output.open("x") as f:
        json.dump(p, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
