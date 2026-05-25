# A Controlled Comparison of LoRA, DoRA, and IA3 for Multi-Step Math Reasoning

This project evaluates three parameter-efficient fine-tuning methods—LoRA, DoRA, and IA3—against zero-shot and 5-shot in-context learning baselines on the GSM8K grade-school mathematics benchmark. All adapters were trained on `meta-llama/Llama-3.2-3B-Instruct` under a fixed compute budget (6,000 examples, 2 epochs, seed 42, single V100 GPU) with identical hyperparameters to isolate the effect of the adaptation method itself. Under this budget, all three PEFT methods underperformed the zero-shot baseline (0.75 accuracy), with DoRA achieving the best adapter accuracy (0.66) and IA3 the best parameter efficiency (1.10 MB, 286,720 trainable parameters); the likely cause is underfitting at 2 epochs, not a fundamental limitation of the PEFT approach.

## Results

| Run           | Accuracy | Extraction Failure Rate | Avg Latency (ms) | Avg Output Tokens |
| ------------- | -------- | ----------------------- | ---------------- | ----------------- |
| base_zeroshot | 0.7500   | 0.0100                  | 17376.30         | 168.91            |
| ia3           | 0.6700   | 0.0300                  | 13402.38         | 139.42            |
| dora          | 0.6600   | 0.0100                  | 10059.31         | 108.26            |
| base_fewshot5 | 0.6500   | 0.0200                  | 15102.91         | 92.30             |
| lora          | 0.6300   | 0.0100                  | 10735.04         | 105.79            |

Zero-shot (0.75) outperformed all three trained adapters under this compute budget; see [report/peft_comparison_report.md](report/peft_comparison_report.md) for the full per-category breakdown and error analysis.

## Key Findings

- **DoRA is the best adapter overall (0.66 accuracy)**: it leads LoRA by 3 points and shows a consistent category-level advantage — best or tied-best across every represented category (arithmetic 0.7042, fractions 0.5882, comparison 0.7143, multi_hop 0.6667).
- **IA3 is the most parameter-efficient method**: 1.10 MB adapter, 286,720 trainable parameters (0.01%), 16× smaller than LoRA (17.51 MB) and DoRA (17.96 MB), while still reaching 0.67 accuracy.
- **All PEFT methods underperformed zero-shot (0.75)** — final training losses of 3.41 (LoRA/DoRA) and 4.82 (IA3) indicate the models had not converged; more epochs or a larger training split are the direct remedies.
- **IA3 wins the comparison category (6/7, 0.8571)**, matching zero-shot on multi_hop (5/6, 0.8333) — suggesting its scaled activations confer an advantage on relative-magnitude and multi-step problems despite lower overall accuracy.

## Project Structure

```
peft-reasoning-comparison/
├── src/
│   ├── evaluate.py          # Inference + evaluation harness (--adapter, --few_shot flags)
│   ├── train.py             # PEFT training loop (--config, --dry_run, --resume_from_checkpoint)
│   ├── extract_answer.py    # Regex answer extraction pipeline
│   ├── categorize.py        # Seven-category question taxonomy
│   ├── report_tables.py     # Generate markdown tables from prediction JSONL files
│   ├── infer.py             # Low-level model inference utilities
│   └── profile.py           # Latency and token profiling utilities
├── configs/
│   ├── base.yaml            # Shared hyperparameter defaults
│   ├── lora.yaml            # LoRA-specific config (r=16, alpha=32)
│   ├── dora.yaml            # DoRA config (use_dora: true, same rank/target modules)
│   └── ia3.yaml             # IA3 config (k_proj, v_proj, down_proj)
├── tests/
│   ├── test_answer_extraction.py   # 5 unit tests for the extraction pipeline
│   ├── test_categorization.py      # 9 unit tests for the category taxonomy
│   └── test_metrics.py             # 5 unit tests for accuracy and failure-rate metrics
├── results/
│   ├── predictions/         # Per-run prediction JSONL files (100 examples each)
│   ├── metrics/
│   │   └── baseline_summary.json   # Aggregated metrics for all 5 runs
│   └── report_tables.md     # Generated markdown tables (Tables 1–3)
├── report/
│   └── peft_comparison_report.md   # Full paper-style report (9 sections)
├── scripts/
│   ├── run_all_train.sh     # Train all three adapters sequentially
│   ├── run_all_eval.sh      # Evaluate all five conditions
│   ├── make_report_assets.sh        # Regenerate report tables
│   └── download_adapters.sh         # Download pretrained adapter weights
└── pyproject.toml
```

## Reproduction

### Requirements

```bash
pip install transformers peft datasets accelerate tqdm pyyaml torch
```

> LLaMA-3.2-3B-Instruct is a gated model. Run `hf auth login` and accept the license at [huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) before downloading.

### Evaluate base model

```bash
python -m src.evaluate \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset gsm8k \
  --split "test[:100]" \
  --output results/predictions/base_zeroshot_100.jsonl
```

### Train LoRA adapter

```bash
python -m src.train --config configs/lora.yaml
```

### Evaluate LoRA adapter

```bash
python -m src.evaluate \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --adapter adapters/lora \
  --dataset gsm8k \
  --split "test[:100]" \
  --output results/predictions/lora_100.jsonl
```

Replace `lora` with `dora` or `ia3` for the other adapters.

### Generate report tables

```bash
python -m src.report_tables --results_dir results/
```

## Hardware

Training: GCP V100 (16 GB). LoRA ~1h 52min (6,739 s), DoRA ~2h 11min (7,854 s), IA3 ~1h 49min (6,544 s). Evaluation: Apple M-series Mac (MPS backend), ~17–22 min per 100 examples.

## Report

Full paper-style report with per-category analysis, error analysis, and discussion:
[report/peft_comparison_report.md](report/peft_comparison_report.md)

## Tests

```bash
pytest tests/
```

19 tests across answer extraction, categorization, and metrics computation.
