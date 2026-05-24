"""CLI for running model evaluation on GSM8K and writing predictions as JSONL."""

import argparse
import json
import random
import time
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
    model_id: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    device = _get_device()
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model = model.to(device)

    model.eval()
    model.generation_config.top_p = None
    return model, tokenizer


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


def _run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
) -> tuple[str, int, float]:
    """Return (output_text, num_output_tokens, latency_ms)."""
    prompt: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(input_device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text, int(len(generated_ids)), round(latency_ms, 2)


def _parse_gold_answer(answer_field: str) -> int | None:
    return extract_answer(answer_field).value


def _evaluate_example(
    idx: int,
    example: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    few_shot_examples: list[dict],
) -> dict:
    question: str = example["question"]
    gold_answer = _parse_gold_answer(example["answer"])

    messages = _build_messages(question, few_shot_examples)
    model_output, num_output_tokens, latency_ms = _run_inference(
        model, tokenizer, messages
    )

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
        "num_output_tokens": num_output_tokens,
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
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model: {args.model}")
    model, tokenizer = _load_model_and_tokenizer(args.model)

    if args.adapter is not None:
        from peft import PeftModel

        print(f"Loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    print(f"Loading dataset: {args.dataset} / {args.split}")
    dataset = load_dataset(args.dataset, "main", split=args.split)

    few_shot_examples: list[dict] = []
    if args.few_shot > 0:
        print(f"Loading {args.few_shot} few-shot examples (seed={args.seed})")
        few_shot_examples = _load_few_shot_examples(args.few_shot, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with output_path.open("w") as fout:
        for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
            record = _evaluate_example(
                idx, example, model, tokenizer, few_shot_examples
            )
            record["adapter_path"] = args.adapter
            records.append(record)
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print("\n--- Summary ---")
    _print_summary(records)


if __name__ == "__main__":
    main()
