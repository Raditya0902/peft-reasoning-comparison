# A Controlled Comparison of LoRA, DoRA, and IA3 for Multi-Step Math Reasoning

## Status

**The project stopped after Phase 1 with a negative result on GSM8K.** Instruct zero-shot had the highest observed accuracy in the shared evaluation pipeline. The three LoRA target formats were **not distinguishable** under the preregistered paired accuracy rule. Self-generated reasoning produced longer answers without recovering accuracy. Phase 2 and the follow-up diagnostic run were canceled.

### Same-pipeline FP16 references

The archived adapters and Instruct baselines were evaluated on the full GSM8K test split with the same batched FP16 V100 pipeline. [Reference results and historical drift](results/phase1-v1/references-v100/report_tables_answer_first.md)

| Run | Accuracy | Bootstrap 95% CI | Avg output tokens | Extraction failure rate | Answer-first rate |
| --- | ---: | --- | ---: | ---: | ---: |
| Instruct zero-shot | 0.7422 | [0.7180, 0.7657] | 172.02 | 0.0182 | 0.0000 |
| Rerun IA3 (2 epochs) | 0.7210 | [0.6967, 0.7445] | 151.38 | 0.0144 | 0.0121 |
| Instruct five-shot | 0.7149 | [0.6899, 0.7392] | 91.09 | 0.0045 | 0.0000 |
| Rerun IA3 (10 epochs) | 0.6937 | [0.6687, 0.7180] | 123.32 | 0.0053 | 0.0152 |
| Rerun DoRA | 0.6399 | [0.6149, 0.6664] | 108.25 | 0.0045 | 0.0136 |
| Rerun LoRA | 0.6285 | [0.6027, 0.6550] | 107.18 | 0.0015 | 0.0114 |

### Phase 1 LoRA target-format comparison

The three arms used the same accepted questions, prompt, and training settings. The table reports full-test results. Accuracy and both rates are fractions. The bootstrap intervals reflect question sampling for each run. [Phase 1 results and paired tests](results/phase1-v1/arms-aligned-v100/report_tables.md)

| Run | Accuracy | Bootstrap 95% CI | Avg output tokens | Extraction failure rate | Answer-first rate |
| --- | ---: | --- | ---: | ---: | ---: |
| Raw gold | 0.6126 | [0.5861, 0.6391] | 94.92 | 0.0030 | 0.0000 |
| Gold with calculator annotations stripped | 0.6050 | [0.5792, 0.6315] | 72.43 | 0.0030 | 0.0000 |
| Rejection-sampled self-generated CoT | 0.5883 | [0.5618, 0.6149] | 165.84 | 0.0644 | 0.0000 |

Answer-first counts outputs that begin with `####` after leading whitespace or end within 10 generated tokens.

### Withdrawn or unsupported claims from the original report

- **Evaluation split.** The original methods and reproduction commands described the pilot subset. The main reported scores came from the full GSM8K test split.
- **Underfitting and overfitting.** The archived training and validation loss summaries do not establish underfitting. The IA3 epoch comparison does not establish overfitting or an optimal stopping epoch. The rerun IA3 schedules were not distinguishable by the Holm-adjusted paired test.
- **Latency.** The original tables combined pilot and full-test timing summaries with incomplete hardware and runtime provenance. Their comparative latency claims are withdrawn.
- **Category tables.** The original category tables are superseded by the audit's regenerated tables. Categories use overlapping lexical rules, and empty categories reflect tagger coverage. Claims of a stable category-specific advantage are unsupported.
- **Training and evaluation prompt consistency.** The archived trainers omitted the evaluator's system instructions. The actual historical date headers are unknown. Phase 1 aligned the training and evaluation headers before training.

[Audit and dated amendments](report/phase0_audit.md) · [Corrected historical category tables](results/phase0-v1/report_tables.md#category-scores)

### Unresolved items

- The self-distillation arm's lower accuracy point estimate remains unexplained.
- The trigger for the historical IA3 answer-first behavior remains unresolved.
- No multi-seed results are available.


## Project Structure

```
peft-reasoning-comparison/
├── src/
│   ├── evaluate.py          # Inference + evaluation harness (--adapter, --few_shot flags)
│   ├── train.py             # PEFT training loop (--config, --dry_run, --resume_from_checkpoint)
│   ├── extract_answer.py    # Regex answer extraction pipeline
│   ├── categorize.py        # Seven-category question taxonomy
│   └── report_tables.py     # Generate markdown tables from prediction JSONL files
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
│   ├── predictions/         # Per-run prediction JSONL files
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

These legacy commands use the current code. The audit describes the limits of reproducing the historical runs. Use fresh output paths for any independent reproduction.

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
  --split "test" \
  --output results/reproduction/instruct_zeroshot.jsonl
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
  --split "test" \
  --output results/reproduction/lora.jsonl
```

Replace `lora` with `dora` or `ia3` for the other adapters.

### Generate report tables

```bash
python -m src.report_tables --results_dir results/
```

## Hardware

The original documentation reports V100 and Apple M-series evaluations. See the audit for the limitations of the timing records and the Phase 1 artifacts for the batched FP16 run. [Latency audit](report/phase0_audit.md#2-evidence) · [Phase 1 run artifacts](results/phase1-v1/arms-aligned-v100/)

## Report

The [original report](report/peft_comparison_report.md) is retained as a historical record. The [audit](report/phase0_audit.md) and [Phase 1 results](results/phase1-v1/arms-aligned-v100/report_tables.md) supersede it.

## Tests

```bash
pytest tests/
```
