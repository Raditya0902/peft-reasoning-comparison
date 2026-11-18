#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?Usage: bash scripts/run_phase1_parity.sh NEW_RESULTS_DIRECTORY [BATCH_SIZE] [DTYPE] [PROMPT_DATE]}
batch_size=${2:-8}
dtype=${3:-bfloat16}
prompt_date=${4:-$(LC_ALL=C date -u "+%d %b %Y")}
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"

for shots in 0 5; do
  run=zeroshot
  if [[ "$shots" == 5 ]]; then run=fewshot5; fi
  args=(
    --model meta-llama/Llama-3.2-3B-Instruct
    --dataset gsm8k --split test --seed 42 --require_cuda
    --few_shot "$shots" --dtype "$dtype" --prompt_date "$prompt_date"
  )
  python -m src.evaluate "${args[@]}" --batch_size "$batch_size" --output "$run_dir/${run}_batched.jsonl"
  python -m src.evaluate "${args[@]}" --batch_size 1 --subset_size 128 --output "$run_dir/${run}_serial.jsonl"
done

python -m src.phase1_parity --run_dir "$run_dir"
