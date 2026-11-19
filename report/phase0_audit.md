# Phase 0 checkpoint

## 1. What I did

- Re-extracted and scored all archived predictions. Pairing checks verified IDs, questions, and stored gold integers within each split.
- Generated question-bootstrap intervals, all pairwise exact McNemar tests, and a category table from fresh tags.
- Audited the training objective, saved loss traces, evaluation provenance, and interpretation in the manuscript.
- This checkpoint used CPU-only analysis of saved artifacts. The training and inference recipes are unchanged pending review.

Regenerate the [full test and historical pilot tables](../results/phase0-v1/report_tables.md) from the repository root.

```sh
python -m src.report_tables --results_dir results --output results/phase0-v1/report_tables.md
```

## 2. Evidence

The [generated tables](../results/phase0-v1/report_tables.md) contain overall scores, 10,000-resample percentile 95% confidence intervals, average output tokens, extraction failures, corrected category scores, and paired tests with discordant counts and Holm correction. The intervals resample questions conditional on each saved run. They do not quantify training-seed variability. Tests use the same questions within each split. The historical pilot overlaps the full test set.

| Issue | Finding and evidence |
| --- | --- |
| 1. Evaluation split | **Confirmed.** The manuscript describes and reproduces the 100-item pilot, while the main table reproduces from the six full-test files. The full-test script and README use `test`. [Manuscript methods](peft_comparison_report.md#2-task-and-dataset) · [Reproducibility commands](peft_comparison_report.md#9-reproducibility) · [Evaluation script](../scripts/run_all_eval.sh#L3-L18) |
| 2. Category overlap and zeros | **Overlap confirmed.** The old tagger independently applied each rule. Its archived arithmetic, fraction, sequence, comparison, and color-based tags overlap, with “uncategorized” counted separately in the manuscript's sum. Both old unit-conversion and algebraic tags were empty even on the full set. Retagging detects mixed units of one dimension with a target-unit cue. Explicit algebra cues still yield zero. These are overlapping lexical groups. In particular, sequencing words and colors with multiple numbers are proxies for the legacy “multi_hop” and “distractor_heavy” labels. A zero count only describes rule coverage. [Category rules](../src/categorize.py#L35-L125) · [Generated category table](../results/phase0-v1/report_tables.md#category-scores) |
| 3. Train versus validation loss | **Non-comparable summaries and likely accumulation artifact.** The repository prints the Trainer's run-wide training loss next to a post-training validation loss. Logs span late steps, so an early-only log is ruled out. In the IA3 trace, late training windows remain near four times validation loss, matching the four configured microbatches per optimizer update. Transformers 4.47.1 skips the reporting division when `num_items_in_batch` exists, which is correct only if the model loss already uses the accumulated denominator. A per-microbatch mean on that branch would be counted four times. The 4.47.1 Llama code passes loss kwargs through to the loss function. Repository history mentions 4.47 compatibility, while the exact Transformers/Accelerate build and runtime loss kwargs for the archived runs remain unverified. The scale artifact is the leading diagnosis, and a whole-run average also includes the high-loss warmup. Do not divide the archive by four or infer optimizer behavior from reported magnitudes. [Training metric calls](../src/train.py#L281-L321) · [IA3 trace](../results/profiles/ia3_convergence_train.log#L857-L870) · [Upstream Trainer reporting path](https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/trainer.py#L3685-L3691) |
| 4. Latency | **Pilot timings carried into the full-test report.** Rounding the LoRA, DoRA and IA3 pilot JSONL means to whole milliseconds gives the values in the manuscript efficiency table. The main table uses their full-test means. The README distinguishes MPS pilot and V100 full-test timings, while the report attributes inference to MPS. The JSONLs lack device, precision and software metadata to resolve hardware per run. The regenerated tables omit timing claims. [Conflicting manuscript tables](peft_comparison_report.md#5-results) · [README hardware notes](../README.md#hardware) · [Saved record schema](../src/evaluate.py#L145-L157) |
| 5. Underfitting and overfitting | **Both explanations are unestablished.** The 10-epoch IA3 recipe scores lower on exact match, while its validation cross-entropy decreases from epoch 1 through epoch 10. Changing total epochs also changes the cosine decay and warmup durations, so the two final runs cannot locate the best epoch or isolate extra updates. Validation exact match across checkpoints is missing. [IA3 training config](../configs/ia3.yaml#L16-L34) · [10-epoch config](../configs/ia3_convergence.yaml#L17-L35) · [Epoch 1 log](../results/profiles/ia3_convergence_train.log#L93) · [Epoch 10 log](../results/profiles/ia3_convergence_train.log#L866) |
| 6. Seeds and uncertainty | **One training seed.** The committed generated table already included a normal-approximation binomial interval. The manuscript's table omitted it, and the repository had no paired tests. The new analysis supplies question-bootstrap intervals and exact paired tests for the saved runs. [Original generated table](../results/report_tables.md#table-1--main-results) · [New analysis](../results/phase0-v1/report_tables.md#paired-comparisons) |
| 7. Learning rate fairness | **Confirmed scope limitation.** Every adapter used the same AdamW peak LR. LoRA and DoRA shared rank and projection targets. IA3 used method-specific placement and had no LR selection. The original IA3 paper used a higher LR in its T-Few recipe with Adafactor and a different architecture. That motivates a sweep while leaving the best LR for this setting open. [LoRA config](../configs/lora.yaml#L11-L30) · [DoRA config](../configs/dora.yaml#L11-L30) · [IA3 config](../configs/ia3.yaml#L11-L29) · [IA3 paper](https://arxiv.org/html/2205.05638#S4) |
| 8. Targets, masking, and shorter outputs | **Confirmed inputs and length association.** Training inserts the GSM8K answer verbatim, including calculator annotations present in the cached dataset, and clones the entire sequence into labels. The collator masks padding only. The evaluator adds a system message absent from training. LoRA and DoRA generate fewer tokens than zero-shot and frequently emit calculator markers. Their extraction failure rates are lower, so flagged extraction failures cannot explain their accuracy losses. Length, supervision, and annotation effects have yet to be isolated. [Training formatter](../src/train.py#L192-L207) · [Collator](../src/train.py#L301-L307) · [Evaluation prompt](../src/evaluate.py#L65-L75) · [Archive checks](../results/phase0-v1/report_tables.md#archive-checks) |
| 9. Instruct baseline and ceiling | **Strong baseline with an untested ceiling.** Meta documents SFT and RLHF for Instruct and reports GSM8K with its own prompting protocol. The archive establishes the baseline and adaptation deltas for the tested recipes. It cannot locate a ceiling, infer benchmark exposure, or measure how adaptation changes when starting from the non-Instruct checkpoint. [Meta model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) · [Full-test scores](../results/phase0-v1/report_tables.md#overall-scores) |

## 3. What the evidence supports

- On this benchmark and seed 42, zero-shot beats five-shot and every archived adapter run after Holm correction. The 2-epoch IA3 recipe beats LoRA, DoRA and the 10-epoch IA3 recipe. The paired test does not distinguish LoRA from DoRA or five-shot from 2-epoch IA3. The saved point estimates still reproduce.
- IA3 has the fewest trainable parameters in the recorded configurations. The checked-out final artifacts also support the save-size ordering, but the weights are Git-ignored and the archived JSONLs carry no checkpoint hash. The generated efficiency table uses the archived parameter counts.
- The category results are descriptive. Fraction/percentage-tagged examples show a larger LoRA and DoRA drop than uncategorized examples in the corrected table. A distinct failure mechanism or stable category advantage would require validated tags and additional comparisons. On the historical multi-hop proxy, DoRA ties 2-epoch IA3, while on fractions/percentages LoRA ties DoRA.
- The 2- versus 10-epoch IA3 accuracy comparison survives. The claimed overfitting mechanism, optimal epoch interval, benchmark ceiling, and general PEFT method ranking remain unresolved.

## 4. Surprises and reporting errors

- The pilot JSONLs place IA3 ahead of DoRA. All pilot pairwise comparisons are inconclusive after Holm correction. Pilot-to-full point estimates come from overlapping samples, so the difference does not establish a significant improvement from changing the evaluation set. [Pilot scores](../results/phase0-v1/report_tables.md#historical-pilot-100-questions)
- Current few-shot code already interleaves user and assistant turns. The archived rows omit the rendered prompt and demonstration IDs, so the manuscript's claim about examples prepended inside the user turn cannot be verified from the predictions. [Message construction](../src/evaluate.py#L65-L75)
- The extractor tries ordered answer-marker patterns and has no fallback to the last number. The audit keeps that scoring rule for comparability. A marker followed by a numeric prefix can still be parsed even if the intended answer differs from the integer returned. [Extractor](../src/extract_answer.py#L13-L32)
- The CUDA loader currently selects BF16 regardless of configured dtype. The next run should verify supported precision and memory on the actual V100 and pin the dependency versions. [Training loader](../src/train.py#L170-L187) · [Evaluation loader](../src/evaluate.py#L29-L46)

## 5. Proposed experiments and GPU budget

Use a single V100 16 GB. The planning anchors are approximately two hours per two-epoch training and 80 minutes per serial full-test evaluation. A serial validation pass over 500 questions is roughly half an hour by linear scaling. All budget figures are estimates and should be refreshed from pilot throughput before launching each stage. [Recorded runtimes](../README.md#hardware)

| Stage | Design | Estimated V100 GPU-hours |
| --- | --- | --- |
| Phase 1 smoke and evaluator verification | On CPU, test assistant-only supervision including the end-of-turn token, padding, template alignment and truncation. Check loss or gradients with accumulation factors 1 and 4 on the same effective batch. Log token-weighted NLL as summed shifted cross-entropy divided by the number of supervised targets. Compare train and validation NLL at identical frozen checkpoints using disjoint data and the same aggregation. On GPU, verify precision, memory, and serial versus batched greedy outputs on a fixed validation pilot, including merged versus unmerged IA3 if merge is used. Compare extracted answers, correctness, token lengths and failure counts before using the faster path. | 0.5–1.0 |
| Phase 1 controlled format ablation | Generate two sampled completions per candidate question from the unadapted Instruct model on the training pool only. Keep one correct response per question using a fixed selection rule and store acceptance, prompt and generation metadata. Use the same accepted question IDs in all three LoRA arms, one seed, assistant-only labels and the same prompt and hyperparameters. Compare original gold, stripped gold, and accepted teacher completions. Include a fresh zero-shot reference under the same evaluator, and report paired accuracy, output length, and extraction failures. Compare on validation before a single frozen full-test evaluation per arm. | About 24–28 at historical throughput, including roughly 12.1 for candidate sampling, 6 or more for three trainings, 1.5 for validation, and 5.3 for four full-test evaluations. Pilot the sampling throughput and accepted pool size before committing the full generation budget. |
| Phase 1 optional historical control | Add a full-prompt gold run on the same accepted question IDs to measure masking against the main masked gold arm. If needed, add an assistant-only gold run on the entire training pool to measure the effect of selecting teacher-solvable questions. Use the same backend and template across each comparison. | Roughly 3.8 per additional two-epoch run with validation and test. |
| Phase 2 LR selection and final comparison | Start with Llama-3.2-3B Base and a plain training/evaluation prompt, with a 1B Base fallback if the 3B pilot cannot fit the budget. Compare against the matching unadapted Base under that fixed prompt, with a fixed five-shot reference for prompt sensitivity. Sweep the requested three LRs per method with seed 42 and select by exact match on `train[6000:6500]`, using a predefined lower-LR tie break. Use LoRA and DoRA rank 16 and the same projections. Retain IA3's documented method-specific projections and hold all common settings and its own placement fixed across its LR sweep. Evaluate selected configs on the full test at seeds 42–44, reusing the validated seed-42 checkpoint when identical. Report mean and sample SD across seeds, question-bootstrap intervals that preserve question identity across seeds, and matched-seed paired tests with multiplicity correction. Compare each checkpoint's adaptation delta against its own unadapted baseline. | About 43, from nine sweep trainings and validation decodes, six additional trainings, nine final adapter test decodes, and two baseline test decodes. |
| Phase 2 optional IA3 epoch curve | Use one fixed 5-epoch schedule and evaluate saved epochs 1, 2, 3 and 5 on validation. This holds the schedule fixed across checkpoint comparisons. | About 6.5 including four validation decodes. |
| Phase 3 | Regenerate manuscript tables and update the manuscript and README against the approved evidence. | 0 |

Interpret Phase 1 as a comparison of target formats. Original versus stripped gold isolates removing calculator notation. The self-generated arm also changes wording, length and reasoning style, so a recovery there would support a target-distribution explanation without establishing length as the causal mediator. Matching the accepted question IDs controls question selection across the three arms and restricts inference to teacher-solvable training questions. Record the acceptance and realized training-token budgets to make that scope visible. The Phase 2 comparison estimates performance after validation-tuning Base and contrasts it with the fixed historical Instruct recipe and the Phase 1 ablation. A causal comparison of initialization under the same tuning protocol would also require a matched LR selection and seed protocol on Instruct.

**Phase 0 checkpoint approved.**

## Phase 1 preregistration

Phase 0 was approved. This section supersedes the original Phase 1 and 2 budgets and designs where they differ.

### Parity before training

Use the current Instruct chat rendering, seed 42, greedy generation, and the full official test split for batched zero-shot and five-shot. Save the rendered prompt, unpadded input IDs, and generated IDs through the first EOS for each question. Replay a fixed seed-42 sample of 128 IDs per condition with batch size one, on the same GPU, model revision, dtype, tokenizer and software environment. Freeze the date inserted by the Llama chat template across parity runs. Compare raw completion IDs after checking exact prompt text and input IDs. Require **at least 95% exact token-sequence agreement** in each replay and **at most 1 percentage point absolute full-test accuracy drift** against each archive. The archives contain decoded text and lengths. Raw-ID parity and prompt equality are therefore verified against the serial replay. Report archive decoded-text agreement separately. A failing condition stops Phase 1 before training.

Use a fresh results directory for every attempt.

```sh
bash scripts/run_phase1_parity.sh results/phase1-v1/parity 8 bfloat16
```

After parity, time a batched sampling pilot on training IDs, with two draws per question and the same batch settings intended for full sampling. Send the revised GPU estimate before launching the arms. Estimate candidate sampling from elapsed seconds per candidate question, three final decodes from measured zero-shot batched test time, and training from the historical per-arm time, scaled for accepted-pool size and measured target lengths once available. Report the uncertainty in the training component until the target workload can be profiled. Use an explicitly recorded supported precision if the historical precision is unavailable.

### Matched LoRA arms and decision rules

Generate two teacher completions per question on `train[:6000]` at temperature 0.7 and top-p 0.95 with the same batched generation path. Keep the first completion with a correctly extracted final integer. Hold its accepted question IDs, LoRA hyperparameters, seed, epochs, and assistant-only objective fixed across raw gold (a), gold with calculator annotations stripped (b), and teacher CoT (c). The token-normalized objective uses a sum of shifted assistant-token cross-entropy divided by the supervised token count across the whole accumulation window. Record the distribution of supervised length, including the end-of-turn token, and the realized total token budgets. Keep all settings fixed before any arm sees test results.

For the three within-ablation pairs, use two-sided exact McNemar on test correctness, with Holm adjustment across those three tests. Claim an accuracy effect only when adjusted p is below 0.05 and the absolute gap **exceeds 2 percentage points**. Report all other format comparisons as **not distinguishable**. Report accuracy, 10,000-resample question-bootstrap intervals, average output tokens, extraction failures, and descriptive accuracy/counts for generated lengths of 0–99, 100–149, 150–199 and 200 or more tokens. Generated length can change with the intervention. These strata are descriptive and do not identify a length-mediated causal effect.

Label the archived LoRA and the parity-verified batched Instruct zero-shot as references. Report arm (a) versus archived LoRA separately, with its paired accuracy gap and exact p-value, as an **historical recipe comparison**. The requested shared accepted pool for arm (a) differs from the archived full 6,000, and the archive lacks complete runtime and loss metadata. This contrast combines masking, normalization and pool selection, with possible runtime differences. Isolating masking plus normalization would require an additional full-prompt run on the same accepted pool, with its historical objective normalization reproduced and the same remaining settings. No extra arm is allocated here.

### Revised Phase 2 plan

Use Llama-3.2-3B Base where feasible on the V100, with an answer-format-aware plain prompt shared across the unadapted zero-shot, fixed five-shot, and adapted conditions. Use solved training-pool demonstrations in the five-shot condition. Use one-epoch seed-42 LR sweeps for LoRA and IA3, three validation candidates per method as specified above. Select by exact-match accuracy on `train[6000:6500]`, with ties going to the lower LR. Use the selected LoRA LR for DoRA and disclose the lack of separate DoRA tuning. Retrain the three methods from scratch for the full two epochs at seeds 42, 43 and 44. Keep the full test reserved for the final nine adapters and two reference baselines.

The updated budget will use six one-epoch trainings, six validation decodes, nine two-epoch trainings, and eleven full-test decodes. At historical serial rates this is roughly 42 GPU-hours. Replace the decode terms with measured Phase 1 throughput at the checkpoint, and use measured training time per full run when available. Phase 2 remains pending checkpoint review.

### Run order after access is available

Install the dependencies in `requirements-phase1.txt` with a PyTorch CUDA build that supports the V100. Use the parity-validated precision and prompt date for each test decode. An example for a successful BF16 parity run is below. The timing pilot comes first and only the full candidate manifest is used in training.

```sh
run=results/phase1-v1
python -m src.phase1_targets --parity "$run/parity/parity.json" \
  --batch_size 4 --dtype bfloat16 --subset_size 128 --output "$run/targets_pilot.jsonl"
```

After sending the updated estimate, use the same sampling batch size and omit `--subset_size` to write `$run/targets.jsonl`. Then use the same LoRA config for all arms.

```sh
date_string=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["zeroshot"]["prompt_date"])' "$run/parity/parity.json")
for arm in gold stripped self_generated; do
  adapter="adapters/phase1-v1/$arm"
  python -m src.train --config configs/lora.yaml --require_cuda --precision bfloat16 \
    --accepted_targets "$run/targets.jsonl" --target_format "$arm" \
    --output_dir "$adapter" --metadata "$run/${arm}_training.json"
  python -m src.evaluate --model meta-llama/Llama-3.2-3B-Instruct \
    --adapter "$adapter" --split test --batch_size 8 --require_cuda \
    --dtype bfloat16 --prompt_date "$date_string" --output "$run/${arm}_test.jsonl"
done
python -m src.phase1_results --run_dir "$run" --parity_dir "$run/parity"
```

### Execution status

Billing was restored and the parity suite ran on a new V100. The BF16 attempt was stopped for throughput before calculating a test score. The FP16 run used one frozen prompt date and the same runtime for serial and batched inference. The [BF16 preflight timing](../results/phase1-v1/smoke_bfloat16.metadata.json) and [FP16 preflight timing](../results/phase1-v1/smoke_float16_native.metadata.json) come from the same non-test questions.

**The evaluator gate failed.** Both conditions met the raw token-sequence agreement threshold against their serial replays. Both exceeded the permitted absolute accuracy drift from the archive. This comparison cannot isolate the source of drift because the historical prompts and runtime were incompletely recorded. The rejection-sampling pilot and training arms stayed off. The VM is stopped. [Parity results](../results/phase1-v1/parity-v100/parity.json) · [Runtime versions](../results/phase1-v1/parity-v100/runtime.json) · [VM status](../results/phase1-v1/parity-v100/vm-status.yaml)

### 2026-11-18 amendment to the parity decision

The 1 percentage point archive-drift gate **failed** for both baselines. The batched FP16 path is judged **equivalent to the matched serial replay** under the preregistered exact-token criterion. The historical archive could not be reproduced within 1 percentage point in this run. The historical rendered prompts and full runtime metadata were not saved, so the drift cannot be attributed among hardware, precision, prompt/template date, and other runtime differences. The original gate remains the record of the test. [Gate results](../results/phase1-v1/parity-v100/parity.json) · [Matched replay and historical comparisons](../results/phase1-v1/archive_diagnosis.md)

Proceed by the user's documented deviation. Freeze the same V100 FP16 evaluator, model revision, batch size, and template date for the four adapter reference reruns. Use the already recorded zero-shot and five-shot evaluations from this pipeline. All new comparisons use same-pipeline references, including raw-gold LoRA versus re-evaluated historical-weight LoRA. The old prediction files remain labeled historical results. The raw-gold comparison still combines objective changes with matched-pool selection.

After recording the six comparable references, time a 200-question rejection-sampling pilot with two completions per question. Estimate the full candidate generation, evaluation, and training workloads before requesting approval. Phase 1 arms require separate approval of the estimate. If the three-arm estimate exceeds about 15 GPU-hours, propose the raw-gold and self-generated arms first, with stripped gold conditional on those results.

#### Same-pipeline pre-training checkpoint

All four stored adapter checkpoints were re-evaluated with their file checksums recorded. The frozen zero-shot rendered prompts and input token IDs matched exactly across all adapter references. The completed tables include all six reference conditions, question-bootstrap intervals, Holm-corrected paired tests, and changes from the historical conclusions. The budget uses measured sampling and inference throughput and projects training time from the historical runtimes. The VM is stopped and the boot disk is retained while the experiment awaits approval. The Phase 1 arms and Phase 2 runs remain pending. [Reference report](../results/phase1-v1/references-v100/report_tables.md) · [Budget and VM cost](../results/phase1-v1/references-v100/budget.json) · [VM status](../results/phase1-v1/references-v100/vm-status.yaml)

### 2026-11-18 IA3 epoch-result gate

The FP16 same-pipeline reference is the primary comparison. The older predictions remain historical. Paired full-test records localize the larger 10-epoch gain to questions where the historical run produced very short responses that began directly with the answer marker. Almost all of the net gain also survives after excluding questions affected by a generation cap or extraction failure in either output. The length-stratified comparison describes an output behavior and cannot establish that length caused accuracy. [IA3 lengths and paired diagnostics](../results/phase1-v1/ia3-gate/diagnosis.md) · [Per-question 2-epoch outcomes](../results/phase1-v1/ia3-gate/ia3_per_question.jsonl) · [Per-question 10-epoch outcomes](../results/phase1-v1/ia3-gate/ia3_convergence_per_question.jsonl)

The code current when the historical outputs were committed used serial greedy decoding with a 512-new-token cap, a 2,048-token input cap, and EOS from the model's generation configuration. It loaded BF16 on CUDA or MPS and explicitly cast the merged adapter to BF16. It let the tokenizer template supply the date and saved decoded text instead of raw IDs or stop reasons. The current V100 rerun uses FP16, a pinned model revision, a frozen template date, and left-padded greedy batches with the same length caps. Its EOS set is recorded in the diagnostics. The historical evaluator source documents the intended settings, while the saved historical outputs cannot certify the exact run environment or EOS identity. Hitting the length limit explains little of the gain. [Historical evaluator at 67f6eae](https://github.com/Raditya0902/peft-reasoning-comparison/blob/67f6eae/src/evaluate.py) · [Current evaluator](../src/evaluate.py) · [Rerun runtime](../results/phase1-v1/parity-v100/runtime.json)

The historical JSONLs identify the final adapter directories. The 10-epoch training log reports saving to `adapters/ia3_convergence`. The local final directories are Git-ignored. Both rerun weight hashes match their local final training checkpoints byte for byte, and the local final and checkpoint configurations agree. Both IA3 configs target `k_proj`, `v_proj`, and `down_proj`, with `down_proj` as the sole feedforward module. The archived evaluation rows and training log lack an adapter digest, so exact historical weight identity is unresolved. [10-epoch training log](../results/profiles/ia3_convergence_train.log#L871) · [2-epoch config](../configs/ia3.yaml) · [10-epoch config](../configs/ia3_convergence.yaml) · [Recorded identities and stop-token IDs](../results/phase1-v1/ia3-gate/diagnosis.json)

Withdraw the Phase 0 inference that the 2-epoch run is distinguishable from the 10-epoch run. The same-pipeline point estimate remains higher at two epochs, and the Holm-corrected paired test leaves the schedules indistinguishable. Claims that the longer run demonstrates overfitting or locates an optimal stopping epoch remain withdrawn. The deficit and the shorter outputs in the historical archive describe that evaluation only. [Same-pipeline paired tests](../results/phase1-v1/references-v100/report_tables.md#paired-tests)

The fixed CPU FP16 check loaded the same adapter bytes as the V100 references. It compared batch size eight with serial decoding on the same sampled prompts for each adapter, including both IA3 checkpoints. All input IDs and extracted answers agreed, and the test verified actual left padding and the corresponding attention masks. A small number of raw generated sequences differed. This check found no adapter-loading or padding bug on the tested CPU path. Its sample and hardware limit what it can establish about the full V100 run. [CPU token-level outputs and metadata](../results/phase1-v1/ia3-gate/cpu)

A CPU first-token probe held the current prompts and final 10-epoch weights fixed while changing FP16 to BF16. It produced nearly the same answer-marker onset count in both precisions on the historical short-answer cohort. This limited probe did not reproduce the historical short-answer pattern by changing precision alone. The exact historical date, executed runtime, and weight hash remain unavailable, so the initiating cause of the archive-side behavior is unresolved. [Precision probe](../results/phase1-v1/ia3-gate/precision-probe.json)
