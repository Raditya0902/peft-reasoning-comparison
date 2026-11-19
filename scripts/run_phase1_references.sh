#!/usr/bin/env bash
set -euo pipefail

parity_dir=${1:?Usage: bash scripts/run_phase1_references.sh PARITY_DIR NEW_RUN_DIR ADAPTER_DIR}
run_dir=${2:?Missing run directory}
adapters=${3:?Missing adapter directory}
[[ ! -e "$run_dir" ]]
for name in lora dora ia3 ia3_convergence; do
  test -s "$adapters/$name/adapter_model.safetensors"
  test -s "$adapters/$name/adapter_config.json"
done

read -r dtype batch_size prompt_date < <(
  python - "$parity_dir" <<'PY'
import json
import sys
from pathlib import Path
from src.phase1_parity import approved_for_phase1
p = Path(sys.argv[1])
gate = json.loads((p / "parity.json").read_text())
if not approved_for_phase1(gate, accept_archive_drift=True):
    raise SystemExit("Serial equivalence has not passed.")
run = gate["zeroshot"]
print(run["model_dtype"].removeprefix("torch."), run["batch_size"], run["prompt_date"])
PY
)
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
for name in lora dora ia3 ia3_convergence; do
  python -m src.evaluate --model meta-llama/Llama-3.2-3B-Instruct --dataset gsm8k \
    --adapter "$adapters/$name" --split test --seed 42 --batch_size "$batch_size" \
    --dtype "$dtype" --prompt_date "$prompt_date" --require_cuda --output "$run_dir/${name}_reference.jsonl"
done
python -m src.phase1_reference --parity_dir "$parity_dir" --reference_dir "$run_dir" \
  --output "$run_dir/report_tables.md"
python -m src.phase1_targets --parity "$parity_dir/parity.json" --accept_archive_drift \
  --batch_size 4 --dtype "$dtype" --subset_size 200 --output "$run_dir/targets_pilot.jsonl"
