# Phase 0 GSM8K Reanalysis

Scores are recomputed from saved text with the repository extractor and stored gold integers. Two-sided exact McNemar tests are paired by question and corrected with Holm within each split. Bootstrap intervals describe question-level uncertainty conditional on these runs and seed. The historical pilot overlaps the full test set.

## Full test (1,319 questions)

### Overall scores

| Run             | Correct  | Accuracy | Bootstrap 95% CI | Avg Output Tokens | Extraction Failure Rate |
| --------------- | -------- | -------- | ---------------- | ----------------- | ----------------------- |
| base_zeroshot   | 997/1319 | 0.7559   | [0.7324, 0.7786] | 173.45            | 0.0182                  |
| base_fewshot5   | 927/1319 | 0.7028   | [0.6778, 0.7278] | 89.81             | 0.0023                  |
| ia3             | 920/1319 | 0.6975   | [0.6725, 0.7218] | 146.27            | 0.0205                  |
| lora            | 844/1319 | 0.6399   | [0.6133, 0.6649] | 109.29            | 0.0045                  |
| dora            | 836/1319 | 0.6338   | [0.6080, 0.6596] | 108.36            | 0.0030                  |
| ia3_convergence | 826/1319 | 0.6262   | [0.6005, 0.6520] | 112.08            | 0.0068                  |

### Category scores

1101/1319 questions match at least one lexical tag, 412 match multiple tags, and there are 1585 tag assignments. Uncategorized questions match none of the rules.

| Category              | base_zeroshot    | base_fewshot5    | lora             | dora             | ia3              | ia3_convergence  |
| --------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| arithmetic            | 720/953 (0.7555) | 670/953 (0.7030) | 618/953 (0.6485) | 615/953 (0.6453) | 661/953 (0.6936) | 602/953 (0.6317) |
| fractions_percentages | 285/398 (0.7161) | 267/398 (0.6709) | 226/398 (0.5678) | 226/398 (0.5678) | 248/398 (0.6231) | 223/398 (0.5603) |
| unit_conversion       | 36/49 (0.7347)   | 36/49 (0.7347)   | 37/49 (0.7551)   | 33/49 (0.6735)   | 38/49 (0.7755)   | 32/49 (0.6531)   |
| multi_hop             | 17/31 (0.5484)   | 17/31 (0.5484)   | 14/31 (0.4516)   | 15/31 (0.4839)   | 15/31 (0.4839)   | 12/31 (0.3871)   |
| algebraic             | —                | —                | —                | —                | —                | —                |
| comparison            | 85/109 (0.7798)  | 75/109 (0.6881)  | 66/109 (0.6055)  | 66/109 (0.6055)  | 76/109 (0.6972)  | 66/109 (0.6055)  |
| distractor_heavy      | 34/45 (0.7556)   | 35/45 (0.7778)   | 28/45 (0.6222)   | 28/45 (0.6222)   | 32/45 (0.7111)   | 28/45 (0.6222)   |
| uncategorized         | 174/218 (0.7982) | 156/218 (0.7156) | 143/218 (0.6560) | 142/218 (0.6514) | 164/218 (0.7523) | 143/218 (0.6560) |

### Paired comparisons

| Left          | Right           | Left Only Correct | Right Only Correct | Gap (pp) | Exact p   | Holm p    |
| ------------- | --------------- | ----------------- | ------------------ | -------- | --------- | --------- |
| base_zeroshot | base_fewshot5   | 168               | 98                 | +5.31    | 2.103e-05 | 0.0001262 |
| base_zeroshot | lora            | 248               | 95                 | +11.60   | 6.828e-17 | 8.876e-16 |
| base_zeroshot | dora            | 256               | 95                 | +12.21   | 3.37e-18  | 4.718e-17 |
| base_zeroshot | ia3             | 168               | 91                 | +5.84    | 1.976e-06 | 1.779e-05 |
| base_zeroshot | ia3_convergence | 265               | 94                 | +12.96   | 5.538e-20 | 8.308e-19 |
| base_fewshot5 | lora            | 196               | 113                | +6.29    | 2.71e-06  | 2.168e-05 |
| base_fewshot5 | dora            | 203               | 112                | +6.90    | 3.306e-07 | 3.637e-06 |
| base_fewshot5 | ia3             | 159               | 152                | +0.53    | 0.7337    | 1         |
| base_fewshot5 | ia3_convergence | 223               | 122                | +7.66    | 5.915e-08 | 7.097e-07 |
| lora          | dora            | 50                | 42                 | +0.61    | 0.4657    | 1         |
| lora          | ia3             | 138               | 214                | -5.76    | 6.022e-05 | 0.0003011 |
| lora          | ia3_convergence | 171               | 153                | +1.36    | 0.345     | 1         |
| dora          | ia3             | 142               | 226                | -6.37    | 1.397e-05 | 9.778e-05 |
| dora          | ia3_convergence | 178               | 168                | +0.76    | 0.6286    | 1         |
| ia3           | ia3_convergence | 219               | 125                | +7.13    | 4.551e-07 | 4.551e-06 |

### Archive checks

| Run             | Changed Score Records | Retagged Questions | Outputs With `<<` |
| --------------- | --------------------- | ------------------ | ----------------- |
| base_zeroshot   | 0                     | 49                 | 0                 |
| base_fewshot5   | 0                     | 49                 | 148               |
| lora            | 0                     | 49                 | 1266              |
| dora            | 0                     | 49                 | 1271              |
| ia3             | 0                     | 49                 | 8                 |
| ia3_convergence | 0                     | 49                 | 907               |

## Historical pilot (100 questions)

### Overall scores

| Run           | Correct | Accuracy | Bootstrap 95% CI | Avg Output Tokens | Extraction Failure Rate |
| ------------- | ------- | -------- | ---------------- | ----------------- | ----------------------- |
| base_zeroshot | 75/100  | 0.7500   | [0.6600, 0.8300] | 168.91            | 0.0100                  |
| ia3           | 67/100  | 0.6700   | [0.5800, 0.7600] | 139.42            | 0.0300                  |
| dora          | 66/100  | 0.6600   | [0.5600, 0.7500] | 108.26            | 0.0100                  |
| base_fewshot5 | 65/100  | 0.6500   | [0.5600, 0.7400] | 92.30             | 0.0200                  |
| lora          | 63/100  | 0.6300   | [0.5400, 0.7300] | 105.79            | 0.0100                  |

### Category scores

84/100 questions match at least one lexical tag, 31 match multiple tags, and there are 123 tag assignments. Uncategorized questions match none of the rules.

| Category              | base_zeroshot  | base_fewshot5  | lora           | dora           | ia3            |
| --------------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| arithmetic            | 56/71 (0.7887) | 46/71 (0.6479) | 49/71 (0.6901) | 50/71 (0.7042) | 50/71 (0.7042) |
| fractions_percentages | 23/34 (0.6765) | 21/34 (0.6176) | 19/34 (0.5588) | 20/34 (0.5882) | 19/34 (0.5588) |
| unit_conversion       | 2/2 (1.0000)   | 1/2 (0.5000)   | 2/2 (1.0000)   | 2/2 (1.0000)   | 2/2 (1.0000)   |
| multi_hop             | 5/6 (0.8333)   | 3/6 (0.5000)   | 4/6 (0.6667)   | 4/6 (0.6667)   | 5/6 (0.8333)   |
| algebraic             | —              | —              | —              | —              | —              |
| comparison            | 7/7 (1.0000)   | 4/7 (0.5714)   | 3/7 (0.4286)   | 5/7 (0.7143)   | 6/7 (0.8571)   |
| distractor_heavy      | 2/3 (0.6667)   | 1/3 (0.3333)   | 1/3 (0.3333)   | 1/3 (0.3333)   | 1/3 (0.3333)   |
| uncategorized         | 13/16 (0.8125) | 12/16 (0.7500) | 8/16 (0.5000)  | 9/16 (0.5625)  | 12/16 (0.7500) |

### Paired comparisons

| Left          | Right         | Left Only Correct | Right Only Correct | Gap (pp) | Exact p | Holm p |
| ------------- | ------------- | ----------------- | ------------------ | -------- | ------- | ------ |
| base_zeroshot | base_fewshot5 | 17                | 7                  | +10.00   | 0.06391 | 0.5752 |
| base_zeroshot | lora          | 22                | 10                 | +12.00   | 0.0501  | 0.501  |
| base_zeroshot | dora          | 20                | 11                 | +9.00    | 0.1496  | 1      |
| base_zeroshot | ia3           | 12                | 4                  | +8.00    | 0.07681 | 0.6145 |
| base_fewshot5 | lora          | 14                | 12                 | +2.00    | 0.845   | 1      |
| base_fewshot5 | dora          | 14                | 15                 | -1.00    | 1       | 1      |
| base_fewshot5 | ia3           | 12                | 14                 | -2.00    | 0.845   | 1      |
| lora          | dora          | 4                 | 7                  | -3.00    | 0.5488  | 1      |
| lora          | ia3           | 12                | 16                 | -4.00    | 0.5716  | 1      |
| dora          | ia3           | 16                | 17                 | -1.00    | 1       | 1      |

### Archive checks

| Run           | Changed Score Records | Retagged Questions | Outputs With `<<` |
| ------------- | --------------------- | ------------------ | ----------------- |
| base_zeroshot | 0                     | 2                  | 0                 |
| base_fewshot5 | 0                     | 2                  | 13                |
| lora          | 0                     | 2                  | 96                |
| dora          | 0                     | 2                  | 94                |
| ia3           | 0                     | 2                  | 0                 |

## Parameter efficiency

| Run             | Trainable Params | Trainable Share (Reported) |
| --------------- | ---------------- | -------------------------- |
| lora            | 4,587,520        | 0.14%                      |
| dora            | 4,702,208        | 0.15%                      |
| ia3             | 286,720          | 0.01%                      |
| ia3_convergence | 286,720          | 0.01%                      |
