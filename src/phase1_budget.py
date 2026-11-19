"""Estimate GPU-hours from the measured timing pilot and archived training runtimes."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

import yaml
from transformers import AutoTokenizer

from src.phase1_reference import ADAPTERS
from src.report_tables import _load_jsonl
from src.training_objective import assistant_tokens

ARMS = ("gold", "stripped", "self_generated")
HOUR = 3600


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def estimate(reference_dir: Path, parity_dir: Path, train_history: Path, tokenizer_path: str) -> dict:
    pilot = _read(reference_dir / "targets_pilot.metadata.json")
    rows = _load_jsonl(reference_dir / "targets_pilot.jsonl")
    accepted = [row for row in rows if row["accepted_index"] is not None]
    if len(rows) != pilot["questions"] or len(accepted) != pilot["accepted"] or not accepted:
        raise ValueError("Pilot counts do not match the saved metadata.")
    training_h = {entry["run"]: entry["train_runtime_s"] / HOUR
                  for entry in _read(train_history) if "train_runtime_s" in entry}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    date = pilot["prompt_date"]

    def tokens(row, arm):
        return len(assistant_tokens({"question": row["question"], "answer": row[arm]}, tokenizer, 1024, date)["input_ids"])

    reference_tokens = sum(tokens(row, "gold") for row in rows)
    train_arms = {arm: training_h["lora"] * sum(tokens(row, arm) for row in accepted) / reference_tokens for arm in ARMS}
    eval_h = {run: _read(reference_dir / f"{run}_reference.metadata.json")["wall_seconds"] / HOUR for run in ADAPTERS}
    zero = _read(parity_dir / "zeroshot_batched.metadata.json")
    five = _read(parity_dir / "fewshot5_batched.metadata.json")
    zero_h = zero["wall_seconds"] / HOUR
    five_h = five["wall_seconds"] / HOUR
    eval_one_arm = eval_h["lora"]
    candidates = 6000
    full_sampling_h = pilot["wall_seconds"] * candidates / len(rows) / HOUR
    acceptance = len(accepted) / len(rows)
    val_share = 500 / zero["examples"]
    final_eval_h = mean(eval_h[run] for run in ("lora", "dora", "ia3"))
    phase_2 = {
        "one_epoch_sweep_training": 3 * (training_h["lora"] + training_h["ia3"]) / 2,
        "six_validation_decodes": 6 * val_share * final_eval_h,
        "nine_full_trainings": 3 * sum(training_h[method] for method in ("lora", "dora", "ia3")),
        "nine_adapter_decodes": 9 * final_eval_h,
        "two_base_reference_decodes": zero_h + five_h,
    }
    three_arm_h = full_sampling_h + sum(train_arms.values()) + len(ARMS) * (1 + val_share) * eval_one_arm
    two_arm_h = full_sampling_h + train_arms["gold"] + train_arms["self_generated"] + 2 * (1 + val_share) * eval_one_arm
    return {
        "sampling_pilot": {
            "questions": len(rows), "accepted": len(accepted), "acceptance_rate": acceptance,
            "samples_per_question": pilot["samples_per_question"],
            "measured_seconds": pilot["wall_seconds"],
            "projected_questions": candidates,
            "projected_samples": candidates * pilot["samples_per_question"],
            "projected_accepted": candidates * acceptance,
            "projected_full_sampling_gpu_hours": full_sampling_h,
        },
        "phase1_remaining": {
            "estimated_training_gpu_hours_per_arm": train_arms,
            "estimated_validation_and_test_gpu_hours_per_arm": (1 + val_share) * eval_one_arm,
            "estimated_gpu_hours": three_arm_h,
            "estimated_gpu_hours_raw_gold_and_teacher_only": two_arm_h,
        },
        "phase2": {
            "estimated_gpu_hours_by_component": phase_2,
            "estimated_total_gpu_hours": sum(phase_2.values()),
        },
        "assumptions": [
            "Sampling scales with questions at unchanged batch size and draws per question.",
            "Training uses historical two-epoch runtimes. Phase 1 scales the LoRA runtime by projected total input tokens, including the prompt, with observed acceptance and target lengths. Padding, hardware precision and optimizer differences can affect runtime.",
            "Phase 1 evaluation uses measured LoRA reference time and scales validation by question count. Phase 2 uses current Instruct inference times as proxies for Base, which may produce different response lengths.",
            "The totals exclude setup, VM idle time and prior GPU measurements. They include no optional IA3 epoch curve.",
        ],
    }


def phase2_from_completed_arms(run_dir: Path, parity: Path, train_history: Path) -> dict:
    """Replace Phase 2 proxies with the measured Phase 1 training and decoding times."""
    train = {arm: _read(run_dir / f"{arm}_training.json") for arm in ARMS}
    if any(entry["status"] != "completed" for entry in train.values()):
        raise ValueError("Phase 2 estimates require completed Phase 1 arms.")
    history = {row["run"]: row["train_runtime_s"]
               for row in _read(train_history) if "train_runtime_s" in row}
    valid_h = mean(_read(run_dir / f"{arm}_validation.metadata.json")["wall_seconds"] / HOUR for arm in ARMS)
    test_h = mean(_read(run_dir / f"{arm}_test.metadata.json")["wall_seconds"] / HOUR for arm in ARMS)
    lora_h = train["gold"]["train_runtime_s"] / HOUR
    # Scale to the planned full pool using observed examples as the time proxy.
    lora_h *= train["gold"]["candidate_count"] / train["gold"]["target_length_tokens"]["count"]
    two_epoch_hours = {method: lora_h * history[method] / history["lora"] for method in ("lora", "dora", "ia3")}
    base_h = sum(_read(parity / f"{mode}_batched.metadata.json")["wall_seconds"] / HOUR
                 for mode in ("zeroshot", "fewshot5"))
    work = {
        "six_one_epoch_sweep_trainings": 1.5 * (two_epoch_hours["lora"] + two_epoch_hours["ia3"]),
        "six_validation_decodes": 6 * valid_h,
        "nine_two_epoch_trainings": 3 * sum(two_epoch_hours.values()),
        "nine_adapter_test_decodes": 9 * test_h,
        "two_base_test_decodes": base_h,
    }
    return {
        "phase1_measured_training_gpu_hours_by_arm": {
            arm: entry["train_runtime_s"] / HOUR for arm, entry in train.items(),
        },
        "phase1_mean_validation_decode_gpu_hours": valid_h,
        "phase1_mean_test_decode_gpu_hours": test_h,
        "phase2_estimated_gpu_hours_by_component": work,
        "phase2_estimated_gpu_hours": sum(work.values()),
        "assumptions": [
            "Scale measured aligned gold training time from the matched subset to the full candidate count.",
            "Scale DoRA and IA3 using historical relative training times.",
            "Use Phase 1 adapter decodes and batched Instruct baselines as proxies for Base.",
            "Exclude setup, idle time, and the optional IA3 epoch curve.",
        ],
    }


def vm_cost(lifecycles: list[dict], disk: dict) -> dict:
    """Apply Iowa public on-demand rates to measured instance intervals."""
    rate_v100 = 2.48
    rate_machine = 0.379998
    rate_ipv4 = 0.005
    rate_disk_per_gib = 0.000136986
    if not disk["type"].endswith("/pd-balanced"):
        raise ValueError("This estimate assumes an Iowa balanced persistent disk.")

    def parse(timestamp):
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    sessions = [
        (parse(d["lastStartTimestamp"]), parse(d["lastStopTimestamp"]))
        for d in lifecycles
    ]
    if not sessions or any(stop <= start for start, stop in sessions):
        raise ValueError("Completed VM lifecycles must have positive elapsed time.")
    as_of = max(stop for _, stop in sessions)
    active = [(stop-start).total_seconds()/HOUR for start, stop in sessions]
    disk_hours = (as_of - parse(disk["creationTimestamp"])).total_seconds() / HOUR
    disk_rate = int(disk["sizeGb"]) * rate_disk_per_gib
    vm_rate = rate_v100 + rate_machine + rate_ipv4
    return {
        "as_of": as_of.isoformat(),
        "active_vm_hours_by_session": active,
        "combined_hourly_vm_rate_usd": vm_rate,
        "estimated_vm_cost_usd": sum(active)*vm_rate,
        "estimated_disk_cost_usd": disk_hours*disk_rate,
        "estimated_total_cost_usd": sum(active)*vm_rate + disk_hours*disk_rate,
        "continuing_disk_usd_per_day": 24*disk_rate,
        "excludes": "Credits, discounts, IPv4 free tier, internet egress, and taxes. This is a list-rate estimate, not a billing export.",
        "price_sources": [
            "https://cloud.google.com/products/compute/gpus-pricing",
            "https://cloud.google.com/products/compute/pricing/general-purpose",
            "https://cloud.google.com/vpc/network-pricing",
            "https://cloud.google.com/compute/disks-image-pricing",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--parity_dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train_history", type=Path, default=Path("results/metrics/baseline_summary.json"))
    parser.add_argument("--vm_lifecycle", type=Path, action="append", default=[])
    parser.add_argument("--disk", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = estimate(args.reference_dir, args.parity_dir, args.train_history, args.tokenizer)
    if args.vm_lifecycle and args.disk:
        report["vm_cost"] = vm_cost(
            [yaml.safe_load(path.read_text()) for path in args.vm_lifecycle], _read(args.disk),
        )
        total_time = sum(report["vm_cost"]["active_vm_hours_by_session"])
        report["phase1_total_including_vm_time_to_date_gpu_hours"] = (
            total_time + report["phase1_remaining"]["estimated_gpu_hours"]
        )
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
