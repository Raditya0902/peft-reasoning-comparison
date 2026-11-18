"""Assistant-only labels and token-weighted causal language modeling loss."""

import torch
from torch.nn import functional as F

IGNORE_INDEX = -100


def assistant_tokens(example: dict, tokenizer, max_seq_length: int, prompt_date: str) -> dict:
    prompt = [{"role": "user", "content": example["question"]}]
    complete = [*prompt, {"role": "assistant", "content": example["answer"]}]
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, date_string=prompt_date)
    full = tokenizer.apply_chat_template(complete, tokenize=True, add_generation_prompt=False, date_string=prompt_date)
    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("The chat template does not yield an aligned assistant completion.")
    target_length = len(full) - len(prefix)
    if target_length >= max_seq_length:
        raise ValueError("Assistant target leaves no preceding context within max_seq_length.")
    full = full[-max_seq_length:]
    labels = [IGNORE_INDEX] * (len(full) - target_length) + full[-target_length:]
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def shifted_target_count(labels: torch.Tensor) -> torch.Tensor:
    return labels[..., 1:].ne(IGNORE_INDEX).sum()


def causal_nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets = labels[:, 1:]
    supervised = targets.ne(IGNORE_INDEX)
    return F.cross_entropy(
        logits[:, :-1, :][supervised].float(),
        targets[supervised],
        reduction="sum",
    )
