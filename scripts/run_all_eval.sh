#!/usr/bin/env bash
set -euo pipefail
python -m src.evaluate --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset gsm8k --split "test" \
  --output results/predictions/base_zeroshot_full.jsonl
python -m src.evaluate --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset gsm8k --split "test" --few_shot 5 --seed 42 \
  --output results/predictions/base_fewshot5_full.jsonl
python -m src.evaluate --model meta-llama/Llama-3.2-3B-Instruct \
  --adapter adapters/lora --dataset gsm8k --split "test" \
  --output results/predictions/lora_full.jsonl
python -m src.evaluate --model meta-llama/Llama-3.2-3B-Instruct \
  --adapter adapters/dora --dataset gsm8k --split "test" \
  --output results/predictions/dora_full.jsonl
python -m src.evaluate --model meta-llama/Llama-3.2-3B-Instruct \
  --adapter adapters/ia3 --dataset gsm8k --split "test" \
  --output results/predictions/ia3_full.jsonl
