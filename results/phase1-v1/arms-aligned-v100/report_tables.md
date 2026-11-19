# Phase 1 Instruct ablation

## Full test

| Run                          | Correct  | Accuracy | Bootstrap 95% CI | Avg Output Tokens | Extraction Failure Rate | Answer-first Rate |
| ---------------------------- | -------- | -------- | ---------------- | ----------------- | ----------------------- | ----------------- |
| instruct_zeroshot_reference  | 979/1319 | 0.7422   | [0.7180, 0.7657] | 172.02            | 0.0182                  | 0.0000            |
| same_pipeline_lora_reference | 829/1319 | 0.6285   | [0.6027, 0.6550] | 107.18            | 0.0015                  | 0.0114            |
| gold                         | 808/1319 | 0.6126   | [0.5861, 0.6391] | 94.92             | 0.0030                  | 0.0000            |
| stripped                     | 798/1319 | 0.6050   | [0.5792, 0.6315] | 72.43             | 0.0030                  | 0.0000            |
| self_generated               | 776/1319 | 0.5883   | [0.5618, 0.6149] | 165.84            | 0.0644                  | 0.0000            |

## Matched-pool format comparisons

| Left     | Right          | Left only | Right only | Gap (pp) | Exact p | Holm p | Decision            |
| -------- | -------------- | --------- | ---------- | -------- | ------- | ------ | ------------------- |
| gold     | stripped       | 94        | 84         | +0.76    | 0.5001  | 0.5777 | not distinguishable |
| gold     | self_generated | 208       | 176        | +2.43    | 0.1135  | 0.3406 | not distinguishable |
| stripped | self_generated | 207       | 185        | +1.67    | 0.2888  | 0.5777 | not distinguishable |

Gold vs stripped tests calculator annotation removal together with its change in supervised target length. Comparisons with the sampled teacher change the reasoning source, content, wording, and supervised target length. These are single-seed comparisons on the teacher-solvable subset.

## Same-pipeline LoRA reference comparison

The reference adapter was trained under the historical recipe and re-evaluated on the same prompts and pipeline. The training contrast jointly changes assistant masking, token normalization, training question selection, and system prompt alignment. Historical training runtime and precision provenance are incomplete.

| Left | Right                        | Left only | Right only | Gap (pp) | Exact p |
| ---- | ---------------------------- | --------- | ---------- | -------- | ------- |
| gold | same_pipeline_lora_reference | 142       | 163        | -1.59    | 0.2521  |

## Validation decode

| Run            | Correct | Accuracy | Bootstrap 95% CI | Avg Output Tokens | Extraction Failure Rate | Answer-first Rate |
| -------------- | ------- | -------- | ---------------- | ----------------- | ----------------------- | ----------------- |
| gold           | 355/500 | 0.7100   | [0.6700, 0.7500] | 93.71             | 0.0020                  | 0.0000            |
| stripped       | 353/500 | 0.7060   | [0.6660, 0.7460] | 70.73             | 0.0020                  | 0.0000            |
| self_generated | 336/500 | 0.6720   | [0.6300, 0.7120] | 165.15            | 0.0460                  | 0.0000            |

## Exploratory prompt-date probe

The same fixed 200 test questions are used in all nine evaluations. Only the Today Date line varies. This probe is outside the preregistered comparisons.

| Arm            | Today Date   | N   | Accuracy | Avg output tokens | Answer-first rate |
| -------------- | ------------ | --- | -------- | ----------------- | ----------------- |
| gold           | 18 Nov 2026  | 200 | 0.6150   | 91.73             | 0.0000            |
| gold           | 18 Nov 2025  | 200 | 0.6250   | 91.84             | 0.0000            |
| gold           | No date line | 200 | 0.6400   | 92.35             | 0.0000            |
| stripped       | 18 Nov 2026  | 200 | 0.6350   | 70.42             | 0.0000            |
| stripped       | 18 Nov 2025  | 200 | 0.6300   | 70.48             | 0.0000            |
| stripped       | No date line | 200 | 0.6450   | 70.92             | 0.0000            |
| self_generated | 18 Nov 2026  | 200 | 0.6050   | 159.72            | 0.0000            |
| self_generated | 18 Nov 2025  | 200 | 0.6100   | 158.95            | 0.0000            |
| self_generated | No date line | 200 | 0.6300   | 153.32            | 0.0000            |

## Accuracy by generated token count

Length strata are descriptive because the intervention can change stratum membership.

| Run                          | Tokens  | N    | Correct | Accuracy |
| ---------------------------- | ------- | ---- | ------- | -------- |
| gold                         | 0–99    | 828  | 569     | 0.6872   |
| gold                         | 100–149 | 369  | 189     | 0.5122   |
| gold                         | 150–199 | 90   | 41      | 0.4556   |
| gold                         | 200+    | 32   | 9       | 0.2812   |
| stripped                     | 0–99    | 1087 | 696     | 0.6403   |
| stripped                     | 100–149 | 189  | 86      | 0.4550   |
| stripped                     | 150–199 | 36   | 14      | 0.3889   |
| stripped                     | 200+    | 7    | 2       | 0.2857   |
| self_generated               | 0–99    | 149  | 93      | 0.6242   |
| self_generated               | 100–149 | 499  | 342     | 0.6854   |
| self_generated               | 150–199 | 380  | 225     | 0.5921   |
| self_generated               | 200+    | 291  | 116     | 0.3986   |
| same_pipeline_lora_reference | 0–99    | 682  | 475     | 0.6965   |
| same_pipeline_lora_reference | 100–149 | 414  | 244     | 0.5894   |
| same_pipeline_lora_reference | 150–199 | 149  | 79      | 0.5302   |
| same_pipeline_lora_reference | 200+    | 74   | 31      | 0.4189   |
| instruct_zeroshot_reference  | 0–99    | 112  | 88      | 0.7857   |
| instruct_zeroshot_reference  | 100–149 | 448  | 387     | 0.8638   |
| instruct_zeroshot_reference  | 150–199 | 404  | 308     | 0.7624   |
| instruct_zeroshot_reference  | 200+    | 355  | 196     | 0.5521   |

## Supervised target lengths

| Run            | Accepted | Candidates | Supervised tokens | Mean   | Min  | P10   | P25   | Median | P75   | P90   | Max   |
| -------------- | -------- | ---------- | ----------------- | ------ | ---- | ----- | ----- | ------ | ----- | ----- | ----- |
| gold           | 5539     | 6000       | 548580            | 99.04  | 23.0 | 53.0  | 68.0  | 90.0   | 122.0 | 157.0 | 319.0 |
| stripped       | 5539     | 6000       | 427101            | 77.11  | 22.0 | 39.0  | 51.0  | 70.0   | 96.0  | 124.0 | 275.0 |
| self_generated | 5539     | 6000       | 893603            | 161.33 | 57.0 | 103.0 | 123.5 | 151.0  | 190.0 | 231.0 | 498.0 |
