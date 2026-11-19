"""CLI for running model evaluation on GSM8K and writing predictions as JSONL."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.categorize import categorize
from src.extract_answer import extract_answer

MAX_NEW_TOKENS = 512
MAX_INPUT_TOKENS = 2048  # LLaMA-3.2 supports 128k context; 512 was over-conservative


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model_and_tokenizer(
    model_id: str, dtype_name: str | None = None, require_cuda: bool = False,
    *, device: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    device = device or _get_device()
    if require_cuda and device != "cuda":
        raise RuntimeError("This run requires CUDA.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    dtype = getattr(torch, dtype_name) if dtype_name else (
        torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    )

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to(device)

    model.eval()
    model.generation_config.top_p = None
    return model, tokenizer



def _load_eval_adapter(model, path: str):
    from peft import PeftModel

    weights = Path(path) / "adapter_model.safetensors"
    checksum = hashlib.sha256(weights.read_bytes()).hexdigest() if weights.is_file() else None
    merged = PeftModel.from_pretrained(model, path).merge_and_unload()
    return merged.eval(), checksum


def _load_few_shot_examples(n: int, seed: int) -> list[dict]:
    if n == 0:
        return []
    train_ds = load_dataset("gsm8k", "main", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(train_ds)), n)
    return [
        {"question": train_ds[i]["question"], "answer": train_ds[i]["answer"]}
        for i in indices
    ]


def _build_messages(question: str, few_shot_examples: list[dict]) -> list[dict]:
    system = (
        "Solve the math word problem step by step. "
        "End your response with '#### <number>' where <number> is the final integer answer."
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    for ex in few_shot_examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


@dataclass(frozen=True)
class Generation:
    prompt: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    text: str


def _trim_after_eos(tokens: list[int], eos_ids: set[int]) -> list[int]:
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index + 1]
    return tokens


def _generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    sample: bool = False,
    num_return_sequences: int = 1,
) -> list[Generation]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=True,
    ).to(next(model.parameters()).device)
    generation_settings = {"temperature": 0.7, "top_p": 0.95, "top_k": 0} if sample else {
        "temperature": 1.0
    }
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            **generation_settings,
        )
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    prompt_width = inputs["input_ids"].shape[1]
    completions = outputs[:, prompt_width:].tolist()
    prompt_tokens = [
        tokens[-count:] for tokens, count in zip(
            inputs["input_ids"].tolist(), inputs["attention_mask"].sum(dim=1).tolist(),
            strict=True,
        )
    ]
    generations = []
    for i, completion in enumerate(completions):
        prompt_idx = i // num_return_sequences
        ids = _trim_after_eos(completion, eos_ids)
        generations.append(Generation(
            prompt=prompts[prompt_idx],
            prompt_token_ids=prompt_tokens[prompt_idx],
            output_token_ids=ids,
            text=tokenizer.decode(ids, skip_special_tokens=True),
        ))
    return generations


def _render_prompt(tokenizer, question: str, few_shot_examples: list[dict], prompt_date: str) -> str:
    return tokenizer.apply_chat_template(
        _build_messages(question, few_shot_examples),
        tokenize=False,
        add_generation_prompt=True,
        date_string=prompt_date,
    )


def _score_generation(
    idx: int, example: dict, generation: Generation, latency_ms: float,
) -> dict:
    question = example["question"]
    gold_answer = extract_answer(example["answer"]).value
    model_output = generation.text
    extraction = extract_answer(model_output)
    category_tags = categorize(question)

    correct: bool = (
        not extraction.extraction_failure
        and gold_answer is not None
        and extraction.value == gold_answer
    )
    failure_reason: str | None = (
        "no answer pattern found in model output"
        if extraction.extraction_failure
        else None
    )

    return {
        "id": f"gsm8k-{idx}",
        "question": question,
        "gold_answer": gold_answer,
        "model_output": model_output,
        "predicted_answer": extraction.value,
        "correct": correct,
        "num_output_tokens": len(generation.output_token_ids),
        "rendered_prompt": generation.prompt,
        "prompt_token_ids": generation.prompt_token_ids,
        "output_token_ids": generation.output_token_ids,
        "latency_ms": latency_ms,
        "category_tags": category_tags,
        "extraction_failure": extraction.extraction_failure,
        "failure_reason": failure_reason,
    }


def _print_summary(records: list[dict]) -> None:
    n = len(records)
    if n == 0:
        print("No records to summarize.")
        return
    n_correct = sum(1 for r in records if r["correct"])
    n_failures = sum(1 for r in records if r["extraction_failure"])
    avg_latency = sum(r["latency_ms"] for r in records) / n
    avg_tokens = sum(r["num_output_tokens"] for r in records) / n
    print(f"accuracy: {n_correct / n:.2f}")
    print(f"extraction_failure_rate: {n_failures / n * 100:.1f}%")
    print(f"avg_latency_ms: {avg_latency:.0f}")
    print(f"avg_output_tokens: {avg_tokens:.0f}")
    print(f"total_examples: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a causal LM on GSM8K.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--adapter", default=None, help="Path to PEFT adapter directory")
    parser.add_argument("--dataset", default="gsm8k", help="Dataset name")
    parser.add_argument("--split", required=True, help="Dataset split, e.g. test[:10]")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--few_shot", type=int, default=0, choices=[0, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--subset_size", type=int, help="Sample this many IDs with --seed")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--prompt_date", default=datetime.now(timezone.utc).strftime("%d %b %Y"))
    args = parser.parse_args()
    if args.batch_size < 1 or (args.subset_size is not None and args.subset_size < 1):
        parser.error("batch_size and subset_size must be positive")
    output_path = Path(args.output)
    metadata_path = output_path.with_suffix(".metadata.json")
    if output_path.exists() or metadata_path.exists():
        parser.error("Output or metadata already exists. Choose a new output path.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model: {args.model}")
    model, tokenizer = _load_model_and_tokenizer(args.model, args.dtype, args.require_cuda)

    adapter_sha256 = None
    if args.adapter is not None:
        print(f"Loading adapter: {args.adapter}")
        model, adapter_sha256 = _load_eval_adapter(model, args.adapter)

    print(f"Loading dataset: {args.dataset} / {args.split}")
    dataset = load_dataset(args.dataset, "main", split=args.split)

    few_shot_examples: list[dict] = []
    if args.few_shot > 0:
        print(f"Loading {args.few_shot} few-shot examples (seed={args.seed})")
        few_shot_examples = _load_few_shot_examples(args.few_shot, args.seed)

    indices = list(range(len(dataset)))
    if args.subset_size is not None:
        indices = sorted(random.Random(args.seed).sample(indices, args.subset_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    start = time.perf_counter()
    with output_path.open("x") as fout:
        for offset in tqdm(range(0, len(indices), args.batch_size), desc="Evaluating batches"):
            batch = indices[offset:offset + args.batch_size]
            examples = [dataset[idx] for idx in batch]
            prompts = [_render_prompt(tokenizer, ex["question"], few_shot_examples, args.prompt_date) for ex in examples]
            batch_start = time.perf_counter()
            generations = _generate_batch(model, tokenizer, prompts)
            batch_wall_ms = (time.perf_counter() - batch_start) * 1000
            for idx, example, generation in zip(batch, examples, generations, strict=True):
                record = _score_generation(idx, example, generation, round(batch_wall_ms / len(batch), 2))
                record["adapter_path"] = args.adapter
                record["batch_wall_ms"] = round(batch_wall_ms, 2)
                records.append(record)
                fout.write(json.dumps(record) + "\n")
            fout.flush()

    device = next(model.parameters()).device
    with metadata_path.open("x") as f:
        json.dump({
            **vars(args),
            "adapter_sha256": adapter_sha256,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "dataset_fingerprint": dataset._fingerprint,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "model_dtype": str(model.dtype),
            "versions": {name: version(name) for name in ("torch", "transformers", "datasets")},
            "examples": len(records),
            "wall_seconds": time.perf_counter() - start,
        }, f, indent=2)

    print("\n--- Summary ---")
    _print_summary(records)


if __name__ == "__main__":
    main()
