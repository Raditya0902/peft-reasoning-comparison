# Same-pipeline V100 test references

Question-bootstrap percentile intervals use 10,000 resamples. Paired McNemar tests are two-sided and exact, with Holm adjustment across all 15 comparisons. All current comparisons use the FP16 V100 pipeline. The historical column is for drift diagnosis.

Answer-first counts outputs that begin with `####` after leading whitespace or end within 10 generated tokens. Each output counts at most once. Current token counts include the first EOS when present.

| Run             | Correct  | Accuracy | Bootstrap 95% CI | Avg Output Tokens | Extraction Failure Rate | Answer-first Rate |
| --------------- | -------- | -------- | ---------------- | ----------------- | ----------------------- | ----------------- |
| base_zeroshot   | 979/1319 | 0.7422   | [0.7180, 0.7657] | 172.02            | 0.0182                  | 0.0000            |
| ia3             | 951/1319 | 0.7210   | [0.6967, 0.7445] | 151.38            | 0.0144                  | 0.0121            |
| base_fewshot5   | 943/1319 | 0.7149   | [0.6899, 0.7392] | 91.09             | 0.0045                  | 0.0000            |
| ia3_convergence | 915/1319 | 0.6937   | [0.6687, 0.7180] | 123.32            | 0.0053                  | 0.0152            |
| dora            | 844/1319 | 0.6399   | [0.6149, 0.6664] | 108.25            | 0.0045                  | 0.0136            |
| lora            | 829/1319 | 0.6285   | [0.6027, 0.6550] | 107.18            | 0.0015                  | 0.0114            |

## Paired tests

| Left          | Right           | Left only | Right only | Gap (pp) | Exact p   | Holm p    | Same pipeline       | Historical          |
| ------------- | --------------- | --------- | ---------- | -------- | --------- | --------- | ------------------- | ------------------- |
| base_zeroshot | base_fewshot5   | 159       | 123        | +2.73    | 0.03695   | 0.2217    | not distinguishable | left higher         |
| base_zeroshot | lora            | 260       | 110        | +11.37   | 3.977e-15 | 5.966e-14 | left higher         | left higher         |
| base_zeroshot | dora            | 249       | 114        | +10.24   | 1.115e-12 | 1.56e-11  | left higher         | left higher         |
| base_zeroshot | ia3             | 136       | 108        | +2.12    | 0.08369   | 0.3267    | not distinguishable | left higher         |
| base_zeroshot | ia3_convergence | 181       | 117        | +4.85    | 0.00025   | 0.00175   | left higher         | left higher         |
| base_fewshot5 | lora            | 232       | 118        | +8.64    | 1.107e-09 | 1.328e-08 | left higher         | left higher         |
| base_fewshot5 | dora            | 220       | 121        | +7.51    | 9.126e-08 | 9.126e-07 | left higher         | left higher         |
| base_fewshot5 | ia3             | 143       | 151        | -0.61    | 0.6832    | 0.6832    | not distinguishable | not distinguishable |
| base_fewshot5 | ia3_convergence | 164       | 136        | +2.12    | 0.1189    | 0.3267    | not distinguishable | left higher         |
| lora          | dora            | 25        | 40         | -1.14    | 0.08168   | 0.3267    | not distinguishable | not distinguishable |
| lora          | ia3             | 125       | 247        | -9.25    | 2.457e-10 | 3.195e-09 | right higher        | right higher        |
| lora          | ia3_convergence | 121       | 207        | -6.52    | 2.37e-06  | 2.133e-05 | right higher        | not distinguishable |
| dora          | ia3             | 131       | 238        | -8.11    | 2.761e-08 | 3.037e-07 | right higher        | right higher        |
| dora          | ia3_convergence | 123       | 194        | -5.38    | 7.92e-05  | 0.0006336 | right higher        | not distinguishable |
| ia3           | ia3_convergence | 165       | 129        | +2.73    | 0.04104   | 0.2217    | not distinguishable | left higher         |

## Changes from historical Phase 0

Historical point ranking: base_zeroshot > base_fewshot5 > ia3 > lora > dora > ia3_convergence.

Same-pipeline point ranking: base_zeroshot > ia3 > base_fewshot5 > ia3_convergence > dora > lora.

Pairwise decisions changed for 6/15 comparisons. Headline paired claims changed for 2/5 claims.

| Claim                                              | Historical | Same pipeline | Changed |
| -------------------------------------------------- | ---------- | ------------- | ------- |
| Zero-shot has higher accuracy than every other run | yes        | no            | yes     |
| 2-epoch IA3 has higher accuracy than LoRA and DoRA | yes        | yes           | no      |
| 2-epoch IA3 has higher accuracy than 10-epoch IA3  | yes        | no            | yes     |
| LoRA and DoRA are not distinguishable              | yes        | yes           | no      |
| Five-shot and 2-epoch IA3 are not distinguishable  | yes        | yes           | no      |
