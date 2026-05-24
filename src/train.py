"""Fine-tune a base model using a specified PEFT method and config."""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


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
    lora: LoraTrainConfig
    training: TrainingConfig


def _load_config(path: str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    def _require(d: dict, key: str, section: str) -> Any:
        if key not in d:
            raise ValueError(f"Missing required key '{key}' in config section '{section}'")
        return d[key]

    m = raw.get("model", {})
    d = raw.get("data", {})
    lo = raw.get("lora", {})
    tr = raw.get("training", {})

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
        lora=LoraTrainConfig(
            r=int(_require(lo, "r", "lora")),
            lora_alpha=int(_require(lo, "lora_alpha", "lora")),
            lora_dropout=float(_require(lo, "lora_dropout", "lora")),
            target_modules=list(_require(lo, "target_modules", "lora")),
            bias=str(_require(lo, "bias", "lora")),
            task_type=str(_require(lo, "task_type", "lora")),
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

    device = _get_device()
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model = model.to(device)

    return model, tokenizer


def _format_and_tokenize(
    example: dict, tokenizer: AutoTokenizer, max_seq_length: int
) -> dict:
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokenized = tokenizer(
        text,
        max_length=max_seq_length,
        truncation=True,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


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
    args = parser.parse_args()

    cfg = _load_config(args.config)

    _set_seeds(cfg.training.seed)

    print(f"Loading model: {cfg.model.name}")
    model, tokenizer = _load_model_and_tokenizer(cfg.model.name, cfg.model.dtype)

    peft_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
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

    orig_cols = train_ds.column_names
    train_ds = train_ds.map(
        lambda ex: _format_and_tokenize(ex, tokenizer, cfg.data.max_seq_length),
        remove_columns=orig_cols,
    )
    val_ds = val_ds.map(
        lambda ex: _format_and_tokenize(ex, tokenizer, cfg.data.max_seq_length),
        remove_columns=orig_cols,
    )

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
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
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    metrics = trainer.evaluate()

    print(f"final train loss: {train_result.training_loss:.4f}")
    print(f"final val loss:   {metrics['eval_loss']:.4f}")

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
