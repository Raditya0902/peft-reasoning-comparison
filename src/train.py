"""Fine-tune a base model using a specified PEFT method and config."""

import argparse
import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset
from peft import IA3Config, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)

from src.token_trainer import TokenLossTrainer
from src.training_objective import IGNORE_INDEX, assistant_tokens

TARGET_FORMATS = ("gold", "stripped", "self_generated")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    dtype: str


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    train_split: str
    val_split: str
    seed: int
    max_seq_length: int


@dataclass(frozen=True)
class LoraTrainConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    bias: str
    task_type: str
    use_dora: bool = False


@dataclass(frozen=True)
class Ia3TrainConfig:
    target_modules: list[str]
    feedforward_modules: list[str]
    task_type: str


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    optimizer: str
    save_strategy: str
    eval_strategy: str
    logging_steps: int
    seed: int
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    lora: LoraTrainConfig | None = None
    ia3: Ia3TrainConfig | None = None


def _load_config(path: str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    def _require(d: dict, key: str, section: str) -> Any:
        if key not in d:
            raise ValueError(f"Missing required key '{key}' in config section '{section}'")
        return d[key]

    m = raw.get("model", {})
    d = raw.get("data", {})
    tr = raw.get("training", {})

    lora_cfg: LoraTrainConfig | None = None
    ia3_cfg: Ia3TrainConfig | None = None
    if "lora" in raw:
        lo = raw["lora"]
        lora_cfg = LoraTrainConfig(
            r=int(_require(lo, "r", "lora")),
            lora_alpha=int(_require(lo, "lora_alpha", "lora")),
            lora_dropout=float(_require(lo, "lora_dropout", "lora")),
            target_modules=list(_require(lo, "target_modules", "lora")),
            bias=str(_require(lo, "bias", "lora")),
            task_type=str(_require(lo, "task_type", "lora")),
            use_dora=bool(lo.get("use_dora", False)),
        )
    elif "ia3" in raw:
        ia = raw["ia3"]
        ia3_cfg = Ia3TrainConfig(
            target_modules=list(_require(ia, "target_modules", "ia3")),
            feedforward_modules=list(_require(ia, "feedforward_modules", "ia3")),
            task_type=str(_require(ia, "task_type", "ia3")),
        )
    else:
        raise ValueError("Config must contain either a 'lora:' or 'ia3:' section")

    return TrainConfig(
        model=ModelConfig(
            name=_require(m, "name", "model"),
            dtype=_require(m, "dtype", "model"),
        ),
        data=DataConfig(
            dataset=_require(d, "dataset", "data"),
            train_split=_require(d, "train_split", "data"),
            val_split=_require(d, "val_split", "data"),
            seed=int(_require(d, "seed", "data")),
            max_seq_length=int(_require(d, "max_seq_length", "data")),
        ),
        training=TrainingConfig(
            output_dir=str(_require(tr, "output_dir", "training")),
            num_train_epochs=int(_require(tr, "num_train_epochs", "training")),
            per_device_train_batch_size=int(_require(tr, "per_device_train_batch_size", "training")),
            gradient_accumulation_steps=int(_require(tr, "gradient_accumulation_steps", "training")),
            learning_rate=float(_require(tr, "learning_rate", "training")),
            lr_scheduler_type=str(_require(tr, "lr_scheduler_type", "training")),
            warmup_ratio=float(_require(tr, "warmup_ratio", "training")),
            optimizer=str(_require(tr, "optimizer", "training")),
            save_strategy=str(_require(tr, "save_strategy", "training")),
            eval_strategy=str(_require(tr, "eval_strategy", "training")),
            logging_steps=int(_require(tr, "logging_steps", "training")),
            seed=int(_require(tr, "seed", "training")),
            bf16=bool(_require(tr, "bf16", "training")),
            fp16=bool(_require(tr, "fp16", "training")),
            gradient_checkpointing=bool(_require(tr, "gradient_checkpointing", "training")),
        ),
        lora=lora_cfg,
        ia3=ia3_cfg,
    )


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model_and_tokenizer(
    model_id: str, dtype_str: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    device = _get_device()
    dtype = getattr(torch, dtype_str) if device in ("cuda", "mps") else torch.float32

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to(device)

    return model, tokenizer


def _record_selection(path: Path, dataset, target_format: str):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indices = [row["train_index"] for row in rows]
    if sorted(indices) != list(range(len(dataset))):
        raise ValueError("Target archive must contain the complete candidate pool with unique indices.")
    accepted = []
    for row in sorted(rows, key=lambda row: row["train_index"]):
        original = dataset[row["train_index"]]
        if (row["question"], row["gold"]) != (original["question"], original["answer"]):
            raise ValueError("The target archive does not match the configured training split.")
        if row["accepted_index"] is not None:
            accepted.append({"question": row["question"], "answer": row[target_format]})
    if not accepted:
        raise ValueError("The target archive contains no accepted examples.")
    return Dataset.from_list(accepted)


def _target_length_summary(dataset) -> dict:
    lengths = np.array([sum(label != IGNORE_INDEX for label in row) for row in dataset["labels"]])
    return {
        "count": len(lengths),
        "total_supervised_tokens": int(lengths.sum()),
        "mean": float(lengths.mean()),
        "percentiles": {str(p): float(np.percentile(lengths, p)) for p in (0, 10, 25, 50, 75, 90, 100)},
    }


def _print_trainable_params(model: Any) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune with LoRA from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load model and apply LoRA, print param counts, then exit without training",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from",
    )
    parser.add_argument("--accepted_targets", type=Path)
    parser.add_argument("--target_format", choices=TARGET_FORMATS)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--precision", choices=["float16", "bfloat16"])
    parser.add_argument("--require_cuda", action="store_true")
    args = parser.parse_args()
    if (args.accepted_targets is None) != (args.target_format is None):
        parser.error("--accepted_targets and --target_format must be used together")
    if args.accepted_targets and args.resume_from_checkpoint:
        parser.error("Phase 1 arms must start fresh.")
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("This run requires CUDA.")

    cfg = _load_config(args.config)
    if args.precision:
        cfg = replace(
            cfg,
            model=replace(cfg.model, dtype=args.precision),
            training=replace(
                cfg.training, fp16=args.precision == "float16", bf16=args.precision == "bfloat16",
            ),
        )

    output_dir = args.output_dir or Path(cfg.training.output_dir)
    metadata = args.metadata or output_dir / "training_metadata.json"
    if not args.dry_run and not args.resume_from_checkpoint and output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Choose a fresh output directory.")
    if not args.dry_run and metadata.exists():
        parser.error("Choose a fresh metadata path.")
    _set_seeds(cfg.training.seed)

    print(f"Loading model: {cfg.model.name}")
    model, tokenizer = _load_model_and_tokenizer(cfg.model.name, cfg.model.dtype)

    if cfg.lora is not None:
        peft_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=cfg.lora.target_modules,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
            use_dora=cfg.lora.use_dora,
        )
    else:
        peft_config = IA3Config(
            target_modules=cfg.ia3.target_modules,
            feedforward_modules=cfg.ia3.feedforward_modules,
            task_type=cfg.ia3.task_type,
        )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()

    _print_trainable_params(model)

    if args.dry_run:
        print("Dry run complete.")
        return

    print(
        f"Loading dataset: {cfg.data.dataset} / "
        f"train={cfg.data.train_split}, val={cfg.data.val_split}"
    )
    train_ds = load_dataset(cfg.data.dataset, "main", split=cfg.data.train_split)
    val_ds = load_dataset(cfg.data.dataset, "main", split=cfg.data.val_split)

    candidate_count = len(train_ds)
    prompt_date = datetime.now(timezone.utc).strftime("%d %b %Y")
    if args.accepted_targets:
        target_meta = json.loads(args.accepted_targets.with_suffix(".metadata.json").read_text())
        if target_meta["questions"] != candidate_count or target_meta["dataset_fingerprint"] != train_ds._fingerprint:
            raise ValueError("Target metadata does not match the configured complete candidate pool.")
        prompt_date = target_meta["prompt_date"]
        train_ds = _record_selection(args.accepted_targets, train_ds, args.target_format)
    train_fingerprint = train_ds._fingerprint
    validation_fingerprint = val_ds._fingerprint
    orig_cols = train_ds.column_names
    train_ds = train_ds.map(
        lambda ex: assistant_tokens(ex, tokenizer, cfg.data.max_seq_length, prompt_date),
        remove_columns=orig_cols,
    )
    val_ds = val_ds.map(
        lambda ex: assistant_tokens(ex, tokenizer, cfg.data.max_seq_length, prompt_date),
        remove_columns=val_ds.column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        optim=cfg.training.optimizer,
        save_strategy=cfg.training.save_strategy,
        eval_strategy=cfg.training.eval_strategy,
        logging_steps=cfg.training.logging_steps,
        seed=cfg.training.seed,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        report_to="none",
        average_tokens_across_devices=False,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = TokenLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    device = next(model.parameters()).device
    effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    run_info = {
        "config": asdict(cfg),
        "output_dir": str(output_dir),
        "base_model_revision": getattr(model.config, "_commit_hash", None),
        "train_fingerprint": train_fingerprint,
        "validation_fingerprint": validation_fingerprint,
        "device": str(device),
        "prompt_date": prompt_date,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model_dtype": str(model.dtype),
        "versions": {
            **{name: version(name) for name in ("torch", "transformers", "peft", "accelerate", "datasets")},
            "python": platform.python_version(),
            "torch_cuda": torch.version.cuda,
        },
        "objective": {
            "label_mask": "assistant completion including end of turn",
            "ignore_index": IGNORE_INDEX,
            "shift_labels": True,
            "numerator": "cross_entropy_sum",
            "denominator": "non-ignored shifted targets in the accumulation window",
            "num_items_in_batch_passed_to": "Trainer.compute_loss_func only",
            "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
            "average_tokens_across_devices": training_args.average_tokens_across_devices,
        },
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "accelerator_gradient_accumulation_steps": trainer.accelerator.gradient_accumulation_steps,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "nominal_effective_batch_size": effective_batch,
        "last_effective_batch_size_per_epoch": len(train_ds) % effective_batch or effective_batch,
        "candidate_count": candidate_count,
        "target_format": args.target_format or "gold",
        "targets_sha256": hashlib.sha256(args.accepted_targets.read_bytes()).hexdigest() if args.accepted_targets else None,
        "target_length_tokens": _target_length_summary(train_ds),
        "status": "started",
    }
    metadata.parent.mkdir(parents=True, exist_ok=True)
    with metadata.open("x") as f:
        json.dump(run_info, f, indent=2)
        f.write("\n")

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()

    print(f"train token NLL (online): {trainer.training_token_nll:.4f}")
    print(f"final validation token NLL: {metrics['eval_loss']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    run_info.update(
        status="completed",
        train_runtime_s=train_result.metrics["train_runtime"],
        online_train_token_nll=trainer.training_token_nll,
        validation=metrics,
        log_history=trainer.state.log_history,
    )
    temporary = metadata.with_suffix(".tmp")
    temporary.write_text(json.dumps(run_info, indent=2) + "\n")
    temporary.replace(metadata)
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
