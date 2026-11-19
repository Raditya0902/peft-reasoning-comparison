# V100 parity diagnosis

Serial replay covers only the fixed sampled IDs. Historical and batched scores on those same IDs provide the comparable denominators. Tests are exact two-sided McNemar with unadjusted diagnostic p-values.

| Baseline | Comparison                        | N    | Left accuracy     | Right accuracy    | Gap (pp) | Left only correct | Right only correct | Exact p |
| -------- | --------------------------------- | ---- | ----------------- | ----------------- | -------- | ----------------- | ------------------ | ------- |
| zeroshot | Batched vs historical, full       | 1319 | 979/1319 (74.22%) | 997/1319 (75.59%) | -1.36    | 43                | 61                 | 0.09504 |
| zeroshot | Batched vs historical, replay IDs | 128  | 97/128 (75.78%)   | 100/128 (78.12%)  | -2.34    | 2                 | 5                  | 0.4531  |
| zeroshot | Serial vs historical, replay IDs  | 128  | 97/128 (75.78%)   | 100/128 (78.12%)  | -2.34    | 2                 | 5                  | 0.4531  |
| zeroshot | Batched vs serial, replay IDs     | 128  | 97/128 (75.78%)   | 97/128 (75.78%)   | +0.00    | 1                 | 1                  | 1       |
| fewshot5 | Batched vs historical, full       | 1319 | 943/1319 (71.49%) | 927/1319 (70.28%) | +1.21    | 38                | 22                 | 0.05189 |
| fewshot5 | Batched vs historical, replay IDs | 128  | 100/128 (78.12%)  | 95/128 (74.22%)   | +3.91    | 6                 | 1                  | 0.125   |
| fewshot5 | Serial vs historical, replay IDs  | 128  | 100/128 (78.12%)  | 95/128 (74.22%)   | +3.91    | 6                 | 1                  | 0.125   |
| fewshot5 | Batched vs serial, replay IDs     | 128  | 100/128 (78.12%)  | 100/128 (78.12%)  | +0.00    | 0                 | 0                  | 1       |
