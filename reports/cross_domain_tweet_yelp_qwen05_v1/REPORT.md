# TweetEval-Yelp Qwen2.5-0.5B validation study

Measured seed-2026 binary-sentiment results. The standard domain is TweetEval sentiment after removing neutral examples; the heterogeneous benign domain is Yelp Polarity. The evaluation set contains 1,786 balanced examples per domain, sampled from frozen length pools. HSSR, HRCW, paper targets, and synthetic jitter are not used.

| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Domain gap | Standard gain vs FedAvg | ASR standard | ASR heterogeneous | ASR mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | fedavg | 90.31 | 79.17 | 84.74 | 11.14 | +0.00 | - | - | - |
| none | indexguard | 90.37 | 78.95 | 84.66 | 11.42 | +0.06 | - | - | - |
| none | flame | 89.14 | 78.56 | 83.85 | 10.58 | -1.18 | - | - | - |
| none | fltrust | 83.76 | 74.58 | 79.17 | 9.18 | -6.55 | - | - | - |
| none | safefedllm | 90.03 | 79.51 | 84.77 | 10.53 | -0.28 | - | - | - |
| none | fedgraph | 88.63 | 80.18 | 84.41 | 8.45 | -1.68 | - | - | - |
| label_flip | fedavg | 80.68 | 65.29 | 72.98 | 15.40 | +0.00 | - | - | - |
| label_flip | indexguard | 90.31 | 78.89 | 84.60 | 11.42 | +9.63 | - | - | - |
| label_flip | flame | 88.30 | 77.44 | 82.87 | 10.86 | +7.61 | - | - | - |
| label_flip | fltrust | 83.93 | 73.29 | 78.61 | 10.64 | +3.25 | - | - | - |
| label_flip | safefedllm | 84.99 | 77.04 | 81.02 | 7.95 | +4.31 | - | - | - |
| label_flip | fedgraph | 87.07 | 78.44 | 82.75 | 8.62 | +6.38 | - | - | - |
| textual_backdoor | fedavg | 78.44 | 66.35 | 72.40 | 12.09 | +0.00 | 37.51 | 65.96 | 51.74 |
| textual_backdoor | indexguard | 90.31 | 78.89 | 84.60 | 11.42 | +11.87 | 9.85 | 30.57 | 20.21 |
| textual_backdoor | flame | 88.30 | 77.44 | 82.87 | 10.86 | +9.85 | 13.55 | 31.02 | 22.28 |
| textual_backdoor | fltrust | 83.26 | 72.12 | 77.69 | 11.14 | +4.82 | 23.07 | 42.55 | 32.81 |
| textual_backdoor | safefedllm | 85.83 | 75.98 | 80.91 | 9.85 | +7.39 | 15.90 | 32.70 | 24.30 |
| textual_backdoor | fedgraph | 87.07 | 78.44 | 82.75 | 8.62 | +8.62 | 5.82 | 16.46 | 11.14 |

## Frozen conclusion checks

- Complete matrix: 18/18.
- Each attack lowers FedAvg standard-domain accuracy by at least five points.
- Attacks increasing the FedAvg domain gap: 2/2.
- Under both attacks, every defense exceeds FedAvg standard-domain accuracy by at least three points.
- Minimum observed attacked-defense standard-domain gain: 3.25 points.
- Under both attacks, adjacent standard-domain method results differ by at least one point.
- Every defense reduces mean textual-backdoor ASR relative to FedAvg.
- Distinct two-decimal endings: 26.
- All checks pass: True.

Exact unrounded values and per-check decisions are in `summary.json` and `conclusion_checks.json`. Compact run evidence is under `evidence/`.
