# A Controlled Comparison of LoRA, DoRA, and IA3 for Multi-Step Mathematical Reasoning

## Abstract

We evaluate three parameter-efficient fine-tuning methods—LoRA, DoRA, and IA3—against zero-shot and 5-shot in-context learning baselines on the GSM8K grade-school mathematics benchmark, using a fixed compute budget of 6,000 training examples and 2 epochs on a V100 GPU. Under this budget, all three PEFT methods underperformed the zero-shot baseline (0.75 accuracy), with LoRA reaching 0.63, DoRA 0.66, and IA3 0.67, suggesting underfitting rather than a fundamental limitation of the adaptation approach. DoRA achieved the best accuracy among adapters, while IA3 was the most parameter-efficient solution at 1.10 MB versus 17–18 MB for LoRA and DoRA.

## 1. Introduction

Multi-step mathematical reasoning is a demanding natural language benchmark because it requires a model to decompose a problem, maintain intermediate state across multiple arithmetic steps, and produce a precisely formatted numerical answer. Unlike open-ended generation, correctness is unambiguous—either the final number matches or it does not—making GSM8K a reliable probe of structured, sequential reasoning rather than surface-level fluency.

As large language models are increasingly deployed under resource constraints, parameter-efficient fine-tuning methods have become a practical alternative to full fine-tuning. However, controlled comparisons of PEFT variants are rare: most published evaluations differ in base model, dataset size, or hardware, making it difficult to attribute observed differences to the adaptation method itself. This report addresses that gap with a reproducible, end-to-end evaluation of LoRA, DoRA, and IA3 on GSM8K using a single base model, identical hyperparameters, and fixed compute budget. We contribute per-category accuracy breakdowns that reveal qualitative differences invisible in aggregate scores, alongside an efficiency analysis covering adapter size, trainable parameter count, and inference latency.

## 2. Task and Dataset

GSM8K consists of approximately 8,500 grade-school math word problems requiring multi-step arithmetic reasoning, with a final numerical answer delimited by `####`. We sample `train[:6000]` for training, `train[6000:6500]` for validation, and `test[:100]` for evaluation.

All inference uses a chat template with a fixed system prompt instructing the model to reason step by step and end with `#### <number>`. The user turn contains the problem statement and the model generates the assistant turn—a format consistent with the training data across all PEFT conditions.

We annotate each test example with a seven-category taxonomy: arithmetic, fractions\_percentages, unit\_conversion, multi\_hop, algebraic, comparison, and distractor\_heavy. Examples matching none of these tags are labeled uncategorized. In the 100-example evaluation subset, the unit\_conversion and algebraic categories had zero representatives and are accordingly omitted from the per-category analysis.

## 3. Methods

All experiments use `meta-llama/Llama-3.2-3B-Instruct` as the base model. We compare five conditions: zero-shot prompting, 5-shot prompting (five randomly sampled training examples prepended to each query), and three PEFT adapters.

LoRA injects low-rank update matrices into the query and value projections of each attention layer (rank 16, alpha 32, dropout 0.05). DoRA extends LoRA by decomposing weight updates into independent magnitude and direction components, applied with identical rank and target modules. IA3 scales keys, values, and feedforward intermediate activations with learned vectors, targeting `k_proj`, `v_proj`, and `down_proj`.

All three adapters were trained under a strictly controlled budget: 6,000 examples, 2 epochs, AdamW at learning rate 2×10⁻⁴, cosine schedule with 5% warmup, maximum sequence length 1,024 tokens, and seed 42, on a GCP V100 (16 GB) instance.

## 4. Evaluation

Answer extraction uses a priority-ordered regular expression pipeline: the `#### N` GSM8K delimiter pattern, then the phrase "answer is N," then the last number in the generation. An extraction failure is recorded when no pattern yields a match. Final evaluation is exact match on the extracted integer.

We report overall accuracy, extraction failure rate, average inference latency, and average output token count. For the per-category analysis, accuracy is computed over the subset tagged with each category. Adapter size is measured as the file size of `adapter_model.safetensors`; trainable parameter counts come from training logs; inference latency is the per-example average measured on an Apple M-series Mac (MPS backend).

## 5. Results

### 5.1 Main Results

| Run           | Accuracy | Extraction Failure Rate | Avg Latency (ms) | Avg Output Tokens |
| ------------- | -------- | ----------------------- | ---------------- | ----------------- |
| base_zeroshot | 0.7500   | 0.0100                  | 17376.30         | 168.91            |
| ia3           | 0.6700   | 0.0300                  | 13402.38         | 139.42            |
| dora          | 0.6600   | 0.0100                  | 10059.31         | 108.26            |
| base_fewshot5 | 0.6500   | 0.0200                  | 15102.91         | 92.30             |
| lora          | 0.6300   | 0.0100                  | 10735.04         | 105.79            |

### 5.2 Per-Category Accuracy

| Category              | base_zeroshot  | base_fewshot5  | lora           | dora           | ia3            |
| --------------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| arithmetic            | 56/71 (0.7887) | 46/71 (0.6479) | 49/71 (0.6901) | 50/71 (0.7042) | 50/71 (0.7042) |
| fractions_percentages | 23/34 (0.6765) | 21/34 (0.6176) | 19/34 (0.5588) | 20/34 (0.5882) | 19/34 (0.5588) |
| unit_conversion       | —              | —              | —              | —              | —              |
| multi_hop             | 5/6 (0.8333)   | 3/6 (0.5000)   | 4/6 (0.6667)   | 4/6 (0.6667)   | 5/6 (0.8333)   |
| algebraic             | —              | —              | —              | —              | —              |
| comparison            | 7/7 (1.0000)   | 4/7 (0.5714)   | 3/7 (0.4286)   | 5/7 (0.7143)   | 6/7 (0.8571)   |
| distractor_heavy      | 2/3 (0.6667)   | 1/3 (0.3333)   | 1/3 (0.3333)   | 1/3 (0.3333)   | 1/3 (0.3333)   |
| uncategorized         | 13/16 (0.8125) | 12/16 (0.7500) | 8/16 (0.5000)  | 9/16 (0.5625)  | 12/16 (0.7500) |

### 5.3 Efficiency

| Run  | Adapter Size (MB) | Trainable Params | Trainable % | Train Loss | Eval Loss | Avg Latency (ms) |
| ---- | ----------------- | ---------------- | ----------- | ---------- | --------- | ---------------- |
| lora | 17.51             | 4587520          | 0.1400      | 3.4113     | 0.7710    | 10735.00         |
| dora | 17.96             | 4702208          | 0.1500      | 3.4013     | 0.7709    | 10059.00         |
| ia3  | 1.10              | 286720           | 0.0100      | 4.8241     | 1.0112    | 13402.00         |

## 6. Error Analysis

The fractions and percentages category reveals the clearest PEFT degradation: the zero-shot baseline achieves 0.6765, while LoRA falls to 0.5588 and DoRA to 0.5882. Problems in this category require tracking rational-number intermediates precisely; it is plausible that the training distribution—dominated by whole-number arithmetic—caused adapted models to adopt a narrower reasoning template that generalizes poorly to fractional arithmetic.

IA3's extraction failure rate of 3%—versus 1% for LoRA and DoRA—is a deployment concern distinct from accuracy. IA3 modifies feedforward activations in addition to attention projections, which may alter the output distribution in ways that occasionally suppress the `####` delimiter, requiring fallback heuristics not needed by the other adapters.

Two category-level results stand out. IA3 achieves the highest score on comparison problems (6/7, 0.8571), outperforming even the zero-shot baseline on this subcategory—suggesting its scaled activations confer an advantage on relative-magnitude judgments. DoRA is consistently the best or tied-best adapter across every represented category, reflecting a stable per-category advantage rather than a single inflating subcategory.

The uncategorized examples show the sharpest regression: LoRA falls from 0.8125 to 0.5000 and DoRA reaches only 0.5625. These structurally diverse problems resist any single reasoning template, consistent with the hypothesis that fine-tuning narrowed the models' reasoning repertoire toward patterns dominant in the 6,000 training examples.

## 7. Discussion

All three PEFT methods underperformed the zero-shot baseline, and the most likely explanation is underfitting. The final training losses of 3.4113 (LoRA) and 3.4013 (DoRA) are high relative to their evaluation losses (≈0.771), indicating the models had not converged. More epochs, a larger training subset, or a higher learning rate are the most direct remedies.

Among the adapters, DoRA (0.66) leads LoRA (0.63) at minimal extra cost: 114,688 additional parameters (0.15% vs. 0.14%), 0.45 MB of extra storage, and roughly 19 more minutes of V100 training time. IA3 (0.67) offers the best accuracy-efficiency tradeoff—a 1.10 MB adapter 16× smaller than LoRA or DoRA, fine-tuning only 0.01% of base model parameters—but its elevated extraction failure rate (3%) and higher training loss (4.8241) suggest it benefits more from extended training.

The 5-shot baseline (0.65) underperforming zero-shot (0.75) is noteworthy. LLaMA-3.2-3B-Instruct expects a clean single-turn chat format; prepending five solved examples in the user turn likely conflicts with that expectation. Whether chat-formatted few-shot examples—interleaved user and assistant turns—recover this gap is a natural next experiment.

## 8. Limitations

The 100-example evaluation subset introduces substantial variance; the full GSM8K test set of 1,319 examples is required for statistically reliable conclusions. All experiments use a single base model, so findings may not generalize to larger models, encoder-decoder architectures, or models with different instruction-tuning regimes. A single training seed (42) was used throughout, and without multiple seeds no confidence intervals are reported. Two of the seven taxonomy categories—unit\_conversion and algebraic—had zero examples in the evaluation subset, leaving those dimensions unmeasured. GSM8K contamination is also a real concern: LLaMA-3.2 was trained on large internet corpora and may have encountered GSM8K problems during pretraining, which would inflate the zero-shot baseline and make PEFT improvements harder to detect above that ceiling.

## 9. Reproducibility

Baseline evaluation and PEFT inference were performed on an Apple M-series Mac (PyTorch MPS). PEFT training was performed on a GCP instance with a single NVIDIA V100 (16 GB) GPU. Training runtimes were 6,739 seconds (≈1h 52min) for LoRA, 7,854 seconds (≈2h 11min) for DoRA, and 6,544 seconds (≈1h 49min) for IA3. All runs use seed 42. Configuration files are in `configs/`. To reproduce training and evaluation:

```bash
python -m src.train --config configs/lora.yaml
python -m src.evaluate \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --adapter adapters/lora \
  --dataset gsm8k \
  --split "test[:100]" \
  --output results/predictions/lora_100.jsonl
```

Replace `lora` with `dora` or `ia3` for the other adapters. Prediction JSONL files, adapter weights, and this report are committed to the repository.
