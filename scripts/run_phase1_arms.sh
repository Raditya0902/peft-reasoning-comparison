#!/usr/bin/env bash
set -Eeuo pipefail

# Launch from the project root on the dedicated Phase 1 V100.
manifest=${1:?Usage: bash scripts/run_phase1_arms.sh MANIFEST NEW_RUN_DIR NEW_ADAPTER_DIR}
run_dir=${2:?Missing run directory}
adapter_dir=${3:?Missing adapter directory}
[[ ! -e "$run_dir" && ! -e "$adapter_dir" ]]
mkdir -p "$run_dir"
stop_log="$run_dir/shutdown.log"

finish() {
  code=$?
  trap - EXIT
  set +e
  printf '%s phase1_exit=%s; scheduling VM shutdown in two minutes\n' "$(date -u +%FT%TZ)" "$code" >> "$stop_log"
  sudo -n shutdown -h +2 "Phase 1 completed or failed" >> "$stop_log" 2>&1
  shutdown_code=$?
  printf '%s shutdown_command_exit=%s\n' "$(date -u +%FT%TZ)" "$shutdown_code" >> "$stop_log"
  (( code != 0 )) || code=$shutdown_code
  exit "$code"
}
trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
python - "$manifest" "$run_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from src.phase1_pretrain import check_prompt_headers

path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
assert manifest["status"] == "prompt_gate_passed", "CPU prompt gate must pass."
targets = Path(manifest["accepted_targets"])
for artifact, digest in ((targets, manifest["accepted_targets_sha256"]),
                         (Path(manifest["shared_config"]), manifest["shared_config_sha256"])):
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest, artifact
with targets.open() as f:
    rows = map(json.loads, f)
    example = next(row for row in rows if row["train_index"] == manifest["prompt_gate"]["train_index"])
tokenizer = AutoTokenizer.from_pretrained(
    manifest["shared_config_values"]["model"]["name"], revision=manifest["model_revision"],
)
assert check_prompt_headers(tokenizer, example, manifest["prompt_date"]) == manifest["prompt_gate"]
(Path(sys.argv[2]) / "pretraining_manifest.json").write_bytes(path.read_bytes())
PY
mapfile -t settings < <(python - "$manifest" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    m = json.load(f)
c = m["shared_config_values"]
print(m["shared_config"], m["accepted_targets"], c["model"]["name"], m["model_revision"],
      m["prompt_date"], c["data"]["val_split"], c["training"]["seed"], sep="\n")
PY
)
config=${settings[0]}
targets=${settings[1]}
model=${settings[2]}
revision=${settings[3]}
frozen_date=${settings[4]}
val_split=${settings[5]}
seed=${settings[6]}
alternate_date='18 Nov 2025'

for arm in gold stripped self_generated; do
  printf '%s train %s\n' "$(date -u +%FT%TZ)" "$arm"
  python -m src.train --config "$config" --require_cuda --accepted_targets "$targets" \
    --target_format "$arm" --output_dir "$adapter_dir/$arm" --metadata "$run_dir/${arm}_training.json" \
    > "$run_dir/${arm}_train.log" 2>&1
  base=(--model "$model" --revision "$revision" --adapter "$adapter_dir/$arm" --dataset gsm8k
        --seed "$seed" --batch_size 8 --dtype float16 --require_cuda)
  for mode in validation test frozen alternate no_date; do
    printf '%s evaluate %s %s\n' "$(date -u +%FT%TZ)" "$arm" "$mode"
    split=test
    date=$frozen_date
    suffix=$mode
    extra=()
    case $mode in
      validation) split=$val_split ;;
      frozen|alternate|no_date)
        suffix="probe_$mode"
        extra=(--subset_size 200)
        if [[ $mode == alternate ]]; then date=$alternate_date; fi
        if [[ $mode == no_date ]]; then extra+=(--omit_prompt_date); fi
        ;;
    esac
    python -m src.evaluate "${base[@]}" --split "$split" --prompt_date "$date" "${extra[@]}" \
      --output "$run_dir/${arm}_${suffix}.jsonl" > "$run_dir/${arm}_${suffix}.log" 2>&1
  done
done
