import json
import random

import pytest

from src.phase1_parity import compare


def test_parity_requires_accuracy_and_exact_tokens_with_equal_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr("src.phase1_parity.FULL_SIZE", 100)
    monkeypatch.setattr("src.phase1_parity.REPLAY_SIZE", 20)
    archive, fast, serial = [tmp_path / f"{name}.jsonl" for name in ("archive", "fast", "serial")]
    rows = [
        {
            "id": f"gsm8k-{i}", "question": str(i), "gold_answer": i,
            "correct": True, "model_output": str(i), "rendered_prompt": str(i),
            "prompt_token_ids": [0, i], "output_token_ids": [i, 3],
        }
        for i in range(100)
    ]
    sample = [dict(rows[i]) for i in random.Random(42).sample(range(100), 20)]

    def write(path, records):
        path.write_text("".join(json.dumps(row) + "\n" for row in records))

    for batch_size, path in ((1, serial), (8, fast)):
        path.with_suffix(".metadata.json").write_text(json.dumps(dict(
            model="model", model_revision="revision", dataset_fingerprint="data",
            device="cuda:0", gpu_name="V100", model_dtype="bf16", versions={},
            split="test", few_shot=0, adapter=None, seed=42, batch_size=batch_size,
            wall_seconds=1, prompt_date="18 Nov 2026",
        )))
    write(archive, rows)
    write(serial, sample)
    write(fast, rows)
    assert compare(archive, fast, serial, 0)["passed"]
    rows[0]["correct"] = rows[1]["correct"] = False
    write(fast, rows)
    assert not compare(archive, fast, serial, 0)["passed"]
    rows[0]["correct"] = True
    write(fast, rows)
    assert compare(archive, fast, serial, 0)["passed"]
    sample[0]["output_token_ids"] = [5]
    write(serial, sample)
    assert compare(archive, fast, serial, 0)["passed"]
    sample[1]["output_token_ids"] = [5]
    write(serial, sample)
    assert not compare(archive, fast, serial, 0)["passed"]
    sample[1]["rendered_prompt"] = "different prompt"
    write(serial, sample)
    with pytest.raises(ValueError, match="prompts differ"):
        compare(archive, fast, serial, 0)


def test_assistant_mask_keeps_target_and_preceding_context():
    pytest.importorskip("torch")
    from src.training_objective import assistant_tokens

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, date_string):
            return [1, 2, 3, 4] if add_generation_prompt else [1, 2, 3, 4, 5, 6, 7]

    row = assistant_tokens({"question": "Q", "answer": "A"}, Tokenizer(), 5, "18 Nov 2026")
    assert row["input_ids"] == [3, 4, 5, 6, 7]
    assert row["labels"] == [-100, -100, 5, 6, 7]
    assert row["attention_mask"] == [1] * 5
    with pytest.raises(ValueError, match="no preceding context"):
        assistant_tokens({"question": "Q", "answer": "A"}, Tokenizer(), 3, "18 Nov 2026")


def test_accumulated_gradient_matches_effective_batch_at_unequal_lengths():
    torch = pytest.importorskip("torch")
    from src.training_objective import causal_nll_sum, shifted_target_count

    logits = torch.randn(4, 4, 6, generator=torch.Generator().manual_seed(42), requires_grad=True)
    labels = torch.tensor([
        [-100, 1, 2, 3],
        [-100, -100, 4, 5],
        [-100, 0, -100, -100],
        [-100, -100, -100, 2],
    ])
    count = shifted_target_count(labels)
    batch_loss = causal_nll_sum(logits, labels) / count
    batch_gradient = torch.autograd.grad(batch_loss, logits)[0]
    micro_losses = [
        causal_nll_sum(logits[i:i + 1], labels[i:i + 1]) / count for i in range(4)
    ]
    accumulated_gradient = sum(torch.autograd.grad(loss, logits)[0] for loss in micro_losses)
    torch.testing.assert_close(accumulated_gradient, batch_gradient)
    torch.testing.assert_close(sum(micro_losses), batch_loss)


def test_pinned_trainer_logs_same_loss_and_update_across_accumulation(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.6":
        pytest.skip("Integration test requires the Phase 1 Transformers version")
    pytest.importorskip("accelerate")
    from transformers.modeling_outputs import CausalLMOutput

    from src.token_trainer import TokenLossTrainer
    from src.training_objective import causal_nll_sum, shifted_target_count

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(42)
            self.weights = torch.nn.Parameter(torch.randn(8, 8, generator=generator))

        def forward(self, input_ids, labels=None):
            return CausalLMOutput(logits=self.weights[input_ids])

    labels = torch.tensor([
        [-100, 1, 2, 3],
        [-100, -100, 4, 5],
        [-100, 0, -100, -100],
        [-100, -100, -100, 2],
    ])
    input_ids = torch.arange(4).repeat(4, 1)
    data = [{"input_ids": ids, "labels": row} for ids, row in zip(input_ids, labels, strict=True)]

    metrics = []
    for microbatch, accumulation in ((4, 1), (1, 4)):
        model = TinyLM()
        args = transformers.TrainingArguments(
            output_dir=tmp_path / str(accumulation), max_steps=1, use_cpu=True,
            per_device_train_batch_size=microbatch, gradient_accumulation_steps=accumulation,
            per_device_eval_batch_size=3, save_strategy="no", report_to="none",
            logging_steps=1, disable_tqdm=True, average_tokens_across_devices=False,
        )
        trainer = TokenLossTrainer(model=model, args=args, train_dataset=data, eval_dataset=data)
        before = (causal_nll_sum(model(input_ids).logits, labels) / shifted_target_count(labels)).item()
        trainer.train()
        assert trainer.training_token_nll == pytest.approx(before)
        assert trainer.state.log_history[0]["loss"] == pytest.approx(before)
        expected_eval = (causal_nll_sum(model(input_ids).logits, labels) / shifted_target_count(labels)).item()
        assert trainer.evaluate()["eval_loss"] == pytest.approx(expected_eval)
        metrics.append(model.weights.detach().clone())
    torch.testing.assert_close(metrics[0], metrics[1])


def test_batched_generation_keeps_eos_but_removes_padding():
    torch = pytest.importorskip("torch")
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    from types import SimpleNamespace
    from transformers import BatchEncoding

    from src.evaluate import _generate_batch

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, prompts, **kwargs):
            return BatchEncoding({
                "input_ids": torch.tensor([[0, 2], [5, 6]]),
                "attention_mask": torch.tensor([[0, 1], [1, 1]]),
            })

        def decode(self, ids, skip_special_tokens):
            return str(ids)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = SimpleNamespace(eos_token_id=[0, 9])

        def generate(self, input_ids, attention_mask, **kwargs):
            return torch.cat((input_ids, torch.tensor([[7, 9, 0], [7, 8, 0]])), dim=1)

    batch = _generate_batch(Model(), Tokenizer(), ["short", "long"])
    assert [row.prompt_token_ids for row in batch] == [[2], [5, 6]]
    assert [row.output_token_ids for row in batch] == [[7, 9], [7, 8, 0]]


def test_rejection_sampling_keeps_first_correct_draw_and_strips_only_annotations():
    pytest.importorskip("torch")
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    from src.evaluate import Generation
    from src.phase1_targets import _candidate_row

    gold = {"question": "Q", "answer": "2 + 2 = <<2+2=4>>4\n#### 4"}
    def make(text):
        return Generation("prompt", [1], [2, 3], text)

    row = _candidate_row(0, gold, [make("#### 5"), make("#### 4")])
    assert row["accepted_index"] == 1
    assert row["stripped"] == "2 + 2 = 4\n#### 4"
    assert _candidate_row(0, gold, [make("#### 4"), make("Correct. #### 4")])["accepted_index"] == 0
    assert _candidate_row(0, gold, [make("#### 5"), make("No match")])["self_generated"] is None


def test_archive_drift_requires_explicit_approval_and_serial_equivalence():
    from src.phase1_parity import approved_for_phase1

    gate = {
        "zeroshot": {"passed": False, "replay_exact_token_sequence_rate": 0.953125},
        "fewshot5": {"passed": False, "replay_exact_token_sequence_rate": 0.984375},
    }
    assert not approved_for_phase1(gate)
    assert approved_for_phase1(gate, accept_archive_drift=True)
    gate["fewshot5"]["replay_exact_token_sequence_rate"] = 0.94
    assert not approved_for_phase1(gate, accept_archive_drift=True)


def test_reference_pairs_align_ids_and_validate_question_identity():
    from src.phase1_reference import _pairs

    one = {"id": "one", "question": "First", "gold_answer": 1, "correct": True}
    two = {"id": "two", "question": "Second", "gold_answer": 2, "correct": False}
    assert _pairs([one, two], [{**two, "correct": True}, one]) == (2, 1, 2, 0, 1)
    with pytest.raises(ValueError, match="Unmatched question"):
        _pairs([one, two], [one, {**two, "question": "Different"}])
