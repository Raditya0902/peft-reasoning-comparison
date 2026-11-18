"""Build a shared training pool for three Phase 1 LoRA target formats."""

import argparse
import json
import random
import re
import time
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.evaluate import _generate_batch, _load_model_and_tokenizer, _render_prompt
from src.extract_answer import extract_answer

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
CANDIDATE_SPLIT = "train[:6000]"
NUM_SAMPLES = 2
SEED = 42
_CALCULATOR = re.compile(r"<<[^<>]*>>")


def _candidate_row(idx, example, generations):
    gold = example["answer"]
    gold_value = extract_answer(gold).value
    if gold_value is None:
        raise ValueError(f"Cannot score gold answer at training index {idx}.")
    candidates = []
    accepted_index = None
    for i, generation in enumerate(generations):
        value = extract_answer(generation.text).value
        correct = value == gold_value
        candidates.append({
            "text": generation.text,
            "num_output_tokens": len(generation.output_token_ids),
            "predicted_answer": value,
            "correct": correct,
        })
        if correct and accepted_index is None:
            accepted_index = i
    return {
        "train_index": idx,
        "question": example["question"],
        "rendered_prompt": generations[0].prompt,
        "gold": gold,
        "stripped": _CALCULATOR.sub("", gold),
        "self_generated": None if accepted_index is None else candidates[accepted_index]["text"],
        "accepted_index": accepted_index,
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], required=True)
    parser.add_argument("--subset_size", type=int, help="Run a timing pilot only")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.subset_size is not None and args.subset_size < 1):
        parser.error("batch_size and subset_size must be positive")
    if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
        parser.error("Choose a new output path.")
    gate = json.loads(args.parity.read_text())
    if not all(gate[name]["passed"] for name in ("zeroshot", "fewshot5")):
        parser.error("Both baseline parity checks must pass.")

    reference = gate["zeroshot"]
    if reference["model_dtype"] != f"torch.{args.dtype}":
        parser.error("Sampling must use the dtype validated by parity.")
    prompt_date = reference["prompt_date"]
    torch.manual_seed(SEED)
    model, tokenizer = _load_model_and_tokenizer(MODEL, args.dtype, require_cuda=True)
    if getattr(model.config, "_commit_hash", None) != reference["model_revision"]:
        parser.error("Sampling must use the model revision validated by parity.")
    dataset = load_dataset("gsm8k", "main", split=CANDIDATE_SPLIT)
    indices = list(range(len(dataset)))
    if args.subset_size is not None:
        indices = sorted(random.Random(SEED).sample(indices, args.subset_size))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    accepted = 0
    with args.output.open("x") as f:
        for offset in tqdm(range(0, len(indices), args.batch_size), desc="Sampling"):
            batch = indices[offset:offset + args.batch_size]
            examples = [dataset[idx] for idx in batch]
            prompts = [_render_prompt(tokenizer, ex["question"], [], prompt_date) for ex in examples]
            generations = _generate_batch(
                model, tokenizer, prompts, sample=True, num_return_sequences=NUM_SAMPLES,
            )
            for i, (idx, example) in enumerate(zip(batch, examples, strict=True)):
                row = _candidate_row(idx, example, generations[i * NUM_SAMPLES:(i + 1) * NUM_SAMPLES])
                accepted += row["accepted_index"] is not None
                f.write(json.dumps(row) + "\n")
            f.flush()

    with args.output.with_suffix(".metadata.json").open("x") as f:
        json.dump({
            "model": MODEL,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "dataset_fingerprint": dataset._fingerprint,
            "split": CANDIDATE_SPLIT,
            "seed": SEED,
            "prompt_date": prompt_date,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "gpu_name": torch.cuda.get_device_name(),
            "samples_per_question": NUM_SAMPLES,
            "questions": len(indices),
            "accepted": accepted,
            "wall_seconds": time.perf_counter() - start,
            "versions": {name: version(name) for name in ("torch", "transformers", "datasets")},
        }, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
