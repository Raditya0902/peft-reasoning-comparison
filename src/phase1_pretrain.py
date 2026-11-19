"""Verify the full matched sampling pool and freeze the Phase 1 run inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer

from src.evaluate import _render_prompt
from src.extract_answer import extract_answer
from src.phase1_targets import _CALCULATOR
from src.prompts import MATH_SYSTEM_INSTRUCTIONS, PHASE1_FROZEN_DATE
from src.report_tables import _load_jsonl
from src.training_objective import IGNORE_INDEX, assistant_tokens, training_prompt_messages

ARMS = ("gold", "stripped", "self_generated")


def _describe(lengths: list[int]) -> dict:
    return {
        "count": len(lengths),
        "total": sum(lengths),
        "mean": float(np.mean(lengths)),
        "sd_population": float(np.std(lengths)),
        "percentiles": {str(p): float(np.percentile(lengths, p)) for p in (0, 5, 25, 50, 75, 95, 100)},
    }


def check_prompt_headers(tokenizer, example: dict, date: str) -> dict:
    """Compare the rendered system blocks with the question and date held fixed."""
    question = example["question"]
    training = tokenizer.apply_chat_template(
        training_prompt_messages(question, instructed=True), tokenize=False,
        add_generation_prompt=True, date_string=date,
    )
    evaluation = _render_prompt(tokenizer, question, [], date)
    system_marker = "<|start_header_id|>system<|end_header_id|>"
    user_marker = "<|start_header_id|>user<|end_header_id|>"

    def header(prompt: str) -> str:
        system, separator, _ = prompt.partition(user_marker)
        if not separator or system_marker not in system:
            raise ValueError("Could not locate the rendered Llama system and user headers.")
        return system

    train_header, eval_header = map(header, (training, evaluation))
    date_line = f"Today Date: {date}"
    def instruction(block: str) -> str:
        return block.split(date_line, 1)[-1].removesuffix("<|eot_id|>").strip()

    train_instruction, eval_instruction = map(instruction, (train_header, eval_header))
    return {
        "train_index": example["train_index"],
        "evaluation_few_shot": 0,
        "prompt_date": date,
        "training_prompt_sha256": hashlib.sha256(training.encode()).hexdigest(),
        "evaluation_prompt_sha256": hashlib.sha256(evaluation.encode()).hexdigest(),
        "training_system_header": train_header,
        "evaluation_system_header": eval_header,
        "training_system_header_sha256": hashlib.sha256(train_header.encode()).hexdigest(),
        "evaluation_system_header_sha256": hashlib.sha256(eval_header.encode()).hexdigest(),
        "training_instruction": train_instruction,
        "evaluation_instruction": eval_instruction,
        "instruction_text_matches": train_instruction == eval_instruction == MATH_SYSTEM_INSTRUCTIONS,
        "system_headers_match": train_header == eval_header,
        "frozen_date_present_in_both": all(date_line in block.splitlines() for block in (train_header, eval_header)),
    }


def build_manifest(targets: Path, config_path: Path, tokenizer_path: str) -> dict:
    metadata = json.loads(targets.with_suffix(".metadata.json").read_text())
    config = yaml.safe_load(config_path.read_text())
    candidate_count = metadata["questions"]
    rows = _load_jsonl(targets)
    if candidate_count != 6000 or [r["train_index"] for r in rows] != list(range(candidate_count)):
        raise ValueError("Expected the complete ordered 6,000-question training manifest.")
    if metadata["split"] != config["data"]["train_split"] or metadata["model_revision"] != config["model"]["revision"]:
        raise ValueError("Sampling split and revision must match the training configuration.")
    if metadata["seed"] != config["training"]["seed"] or metadata["samples_per_question"] != 2:
        raise ValueError("Sampling settings differ from the Phase 1 design.")

    selected = []
    for row in rows:
        gold = extract_answer(row["gold"]).value
        if gold is None or len(row["candidates"]) != metadata["samples_per_question"]:
            raise ValueError(f"Invalid candidate pool at training index {row['train_index']}.")
        candidates = [c["text"] for c in row["candidates"]]
        first = next((i for i, text in enumerate(candidates) if extract_answer(text).value == gold), None)
        teacher = candidates[first] if first is not None else None
        if (row["accepted_index"] != first or row["self_generated"] != teacher
                or row["stripped"] != _CALCULATOR.sub("", row["gold"])):
            raise ValueError(f"Selected target differs from the specified rule at index {row['train_index']}.")
        if first is not None:
            selected.append(row)
    if not selected or len(selected) != metadata["accepted"]:
        raise ValueError("Accepted count differs from sampling metadata.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    max_seq_length = config["data"]["max_seq_length"]
    date = metadata["prompt_date"]
    prompt_gate = check_prompt_headers(tokenizer, selected[0], date)
    teacher_matches = sum(
        row.get("rendered_prompt") == _render_prompt(tokenizer, row["question"], [], PHASE1_FROZEN_DATE)
        for row in rows
    )
    gate_passed = (
        date == PHASE1_FROZEN_DATE
        and all(prompt_gate[field] for field in (
            "system_headers_match", "instruction_text_matches", "frozen_date_present_in_both"
        ))
        and teacher_matches == len(rows)
    )
    arms = {}
    for arm in ARMS:
        encoded = [
            assistant_tokens({"question": row["question"], "answer": row[arm]}, tokenizer, max_seq_length, date, instructed=True)
            for row in selected
        ]
        arms[arm] = {
            "target_format": arm,
            "output_dir": f"adapters/phase1-v1/{arm}",
            "metadata_output": f"results/phase1-v1/{arm}_training.json",
            "supervised_target_tokens": _describe(
                [sum(token != IGNORE_INDEX for token in row["labels"]) for row in encoded]
            ),
            "input_tokens_including_prompt": _describe([len(row["input_ids"]) for row in encoded]),
        }

    teacher_lengths = [row["candidates"][row["accepted_index"]]["num_output_tokens"] for row in selected]
    selected_ids = ",".join(str(row["train_index"]) for row in selected)
    return {
        "status": "prompt_gate_passed" if gate_passed else "blocked_by_prompt_gate",
        "shared_config": str(config_path),
        "shared_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "shared_config_values": config,
        "model_revision": metadata["model_revision"],
        "prompt_date": date,
        "prompt_gate": prompt_gate,
        "rejection_sampling_prompt_verification": {
            "saved_prompt_count": len(rows),
            "exact_matches_to_instructed_render": teacher_matches,
            "prompt_date": PHASE1_FROZEN_DATE,
            "system_instructions": MATH_SYSTEM_INSTRUCTIONS,
            "first_prompt_sha256": hashlib.sha256(rows[0]["rendered_prompt"].encode()).hexdigest(),
        },
        "accepted_targets": str(targets),
        "accepted_targets_sha256": hashlib.sha256(targets.read_bytes()).hexdigest(),
        "accepted_question_ids_sha256": hashlib.sha256(selected_ids.encode()).hexdigest(),
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "candidate_questions": candidate_count,
        "candidates_generated": candidate_count * metadata["samples_per_question"],
        "accepted_questions": len(selected),
        "acceptance_rate": len(selected)/candidate_count,
        "first_correct_draw_index_counts": [
            sum(row["accepted_index"] == i for row in selected) for i in range(metadata["samples_per_question"])
        ],
        "sampling_gpu_hours": metadata["wall_seconds"]/3600,
        "selected_teacher_generated_tokens": _describe(teacher_lengths),
        "effective_train_batch_size": (
            config["training"]["per_device_train_batch_size"] * config["training"]["gradient_accumulation_steps"]
        ),
        "objective": "Assistant-only shifted cross-entropy sum divided by supervised targets in the accumulation window.",
        "arm_settings": arms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.targets, args.config, args.tokenizer)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    if result["status"] != "prompt_gate_passed":
        parser.exit(1, f"Training/evaluation prompt gate failed. See {args.output}.\n")


if __name__ == "__main__":
    main()
