"""Run the evaluator's actual adapter load and greedy generation code on CPU."""

import argparse
import gc
import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import torch

from src.evaluate import MAX_INPUT_TOKENS, MAX_NEW_TOKENS, _generate_batch, _load_eval_adapter, _load_model_and_tokenizer
from src.extract_answer import extract_answer
from src.phase1_reference import ADAPTERS
from src.report_tables import _load_jsonl


def _verify_padding(tokenizer, prompts, unpadded_ids):
    encoded = tokenizer(prompts, truncation=True, max_length=MAX_INPUT_TOKENS, padding=True)
    padding_rows = 0
    for ids, mask, expected in zip(encoded["input_ids"], encoded["attention_mask"], unpadded_ids, strict=True):
        pad = len(mask) - sum(mask)
        if mask != [0] * pad + [1] * len(expected):
            raise ValueError("Attention mask does not describe a left-padded input.")
        if ids[:pad] != [tokenizer.pad_token_id] * pad or ids[pad:] != expected:
            raise ValueError("Left-padded input IDs differ from the frozen unpadded IDs.")
        padding_rows += pad > 0
    return padding_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--zero_shot", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--adapter_dir", type=Path, default=Path("adapters"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference = sorted(_load_jsonl(args.zero_shot), key=lambda r: int(r["id"].split("-")[-1]))
    sample = [reference[i] for i in sorted(random.Random(42).sample(range(len(reference)), 50))]
    batch_size = 8
    for name in ADAPTERS:
        print(f"Loading {name} on CPU", flush=True)
        t0 = time.perf_counter()
        model, tokenizer = _load_model_and_tokenizer(args.model_path, "float16", device="cpu")
        if next(model.parameters()).device.type != "cpu" or tokenizer.padding_side != "left":
            raise ValueError("Expected CPU inference and left tokenizer padding.")
        model, checksum = _load_eval_adapter(model, str(args.adapter_dir / name))
        metadata = json.loads((args.reference_dir / f"{name}_reference.metadata.json").read_text())
        if checksum != metadata["adapter_sha256"]:
            raise ValueError(f"{name} does not match the V100 checkpoint.")
        exact_ids = exact_output = exact_answer = first_token_matches = 0
        padding_rows = 0
        with (args.output_dir / f"{name}.jsonl").open("x") as f:
            for offset in range(0, len(sample), batch_size):
                rows = sample[offset:offset + batch_size]
                prompts = [row["rendered_prompt"] for row in rows]
                expected_ids = [row["prompt_token_ids"] for row in rows]
                padding_rows += _verify_padding(tokenizer, prompts, expected_ids)
                fast = _generate_batch(model, tokenizer, prompts)
                for row, batched in zip(rows, fast, strict=True):
                    serial = _generate_batch(model, tokenizer, [row["rendered_prompt"]])[0]
                    ids_match = batched.prompt_token_ids == serial.prompt_token_ids == row["prompt_token_ids"]
                    exact_ids += ids_match
                    if not ids_match:
                        raise ValueError(f"Input token IDs changed for {name}, {row['id']}.")
                    exact_output += batched.output_token_ids == serial.output_token_ids
                    first_token_matches += batched.output_token_ids[:1] == serial.output_token_ids[:1]
                    answers = [extract_answer(g.text).value for g in (batched, serial)]
                    exact_answer += answers[0] == answers[1]
                    f.write(json.dumps({
                        "id": row["id"],
                        "prompt_token_ids_equal": ids_match,
                        "batched_output_token_ids": batched.output_token_ids,
                        "serial_output_token_ids": serial.output_token_ids,
                        "batched_answer": answers[0],
                        "serial_answer": answers[1],
                    }) + "\n")
                    f.flush()
                print(f"{name}: {offset + len(rows)}/{len(sample)} comparisons in {time.perf_counter()-t0:.1f}s", flush=True)
        result = {
            "device": "cpu",
            "dtype": str(model.dtype),
            "padding_side": tokenizer.padding_side,
            "padding_rows": padding_rows,
            "batch_size": batch_size,
            "examples": len(sample),
            "input_ids_exact": exact_ids,
            "output_ids_exact": exact_output,
            "first_output_id_exact": first_token_matches,
            "extracted_answer_exact": exact_answer,
            "adapter_sha256": checksum,
            "max_new_tokens": MAX_NEW_TOKENS,
            "wall_seconds": time.perf_counter() - t0,
            "versions": {key: version(key) for key in ("torch", "transformers", "peft")},
        }
        (args.output_dir / f"{name}.metadata.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        del model, tokenizer
        gc.collect()


if __name__ == "__main__":
    main()
