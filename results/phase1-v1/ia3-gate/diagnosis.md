# IA3 archive-drift diagnosis

## Lengths, extraction, and generation stops

Recorded lengths include EOS when present. Historical cap hits are upper bounds on truncation because the historical JSONL omits raw output IDs. In the rerun, a capped output without EOS is confirmed truncated.

| Run             | Source        | Accuracy | Mean   | Min | P5 | P25 | Median | P75 | P95 | Max | Extraction failures | Cap hits       | Confirmed truncation |
| --------------- | ------------- | -------- | ------ | --- | -- | --- | ------ | --- | --- | --- | ------------------- | -------------- | -------------------- |
| ia3             | Same pipeline | 72.10%   | 151.38 | 4   | 78 | 111 | 138    | 179 | 265 | 512 | 19/1319 (1.44%)     | 8/1319 (0.61%) | 8                    |
| ia3             | Historical    | 69.75%   | 146.27 | 4   | 68 | 106 | 134    | 177 | 278 | 512 | 27/1319 (2.05%)     | 5/1319 (0.38%) | unknown              |
| ia3_convergence | Same pipeline | 69.37%   | 123.32 | 4   | 59 | 87  | 113    | 150 | 217 | 512 | 7/1319 (0.53%)      | 4/1319 (0.30%) | 4                    |
| ia3_convergence | Historical    | 62.62%   | 112.08 | 4   | 4  | 76  | 105    | 142 | 207 | 512 | 9/1319 (0.68%)      | 7/1319 (0.53%) | unknown              |

## Paired outcomes on identical questions

| Run             | Both correct | Rerun only | Historical only | Both wrong | Different extracted answer | Exact p   |
| --------------- | ------------ | ---------- | --------------- | ---------- | -------------------------- | --------- |
| ia3             | 859          | 92         | 61              | 307        | 269                        | 0.01502   |
| ia3_convergence | 758          | 157        | 68              | 336        | 374                        | 2.788e-09 |

| Run             | Subset                           | Questions | Net rerun correct |
| --------------- | -------------------------------- | --------- | ----------------- |
| ia3             | cap in either run                | 10        | +0                |
| ia3             | extraction failure in either run | 36        | +11               |
| ia3             | neither cap nor failure          | 1283      | +20               |
| ia3             | historical 10 tokens or less     | 51        | +22               |
| ia3             | historical over 10 tokens        | 1268      | +9                |
| ia3_convergence | cap in either run                | 11        | +1                |
| ia3_convergence | extraction failure in either run | 15        | +2                |
| ia3_convergence | neither cap nor failure          | 1304      | +87               |
| ia3_convergence | historical 10 tokens or less     | 106       | +63               |
| ia3_convergence | historical over 10 tokens        | 1213      | +26               |

The last two rows for each run stratify by historical output length. They describe where the gain occurred. The historical length belongs to the generated response and does not identify a causal intervention.

## Adapter identity

The final local weights match the V100 rerun hash. The historical JSONLs record directories and omit checkpoint hashes. The local final checkpoint match establishes the rerun epoch, while it cannot attest to the historical file contents.

| Run             | Historical adapter path  | Local final checkpoint | Rerun SHA-256                                                    | Local checkpoint weights and config match |
| --------------- | ------------------------ | ---------------------- | ---------------------------------------------------------------- | ----------------------------------------- |
| ia3             | adapters/ia3             | checkpoint-750         | 4322ee58f94a86aef2bd18455317c5af6b87082acc845ed3c8b1980d5b6be609 | yes                                       |
| ia3_convergence | adapters/ia3_convergence | checkpoint-3750        | 829c76484646a742de7e0078c2ce1229fadc32f3469ee8ffb3028559116b7075 | yes                                       |
