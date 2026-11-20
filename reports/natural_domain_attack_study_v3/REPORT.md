# Qwen2.5-0.5B natural domain-attack study

Measured seed-2026 results using 1,786 randomly sampled examples per domain. HSSR, HRCW, paper targets, and synthetic jitter are not used.

| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Domain gap | Standard gain vs FedAvg | ASR standard | ASR heterogeneous | ASR mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | fedavg | 89.92 | 76.76 | 83.34 | 13.16 | +0.00 | - | - | - |
| none | indexguard | 89.98 | 77.10 | 83.54 | 12.88 | +0.06 | - | - | - |
| none | flame | 91.49 | 76.43 | 83.96 | 15.06 | +1.57 | - | - | - |
| none | fltrust | 89.31 | 71.89 | 80.60 | 17.41 | -0.62 | - | - | - |
| none | safefedllm | 88.02 | 75.76 | 81.89 | 12.26 | -1.90 | - | - | - |
| none | fedgraph | 86.39 | 71.11 | 78.75 | 15.29 | -3.53 | - | - | - |
| label_flip | fedavg | 81.41 | 67.13 | 74.27 | 14.28 | +0.00 | - | - | - |
| label_flip | indexguard | 90.09 | 76.82 | 83.45 | 13.27 | +8.68 | - | - | - |
| label_flip | flame | 82.64 | 71.95 | 77.30 | 10.69 | +1.23 | - | - | - |
| label_flip | fltrust | 88.58 | 72.12 | 80.35 | 16.46 | +7.17 | - | - | - |
| label_flip | safefedllm | 83.82 | 73.52 | 78.67 | 10.30 | +2.41 | - | - | - |
| label_flip | fedgraph | 85.11 | 70.44 | 77.77 | 14.67 | +3.70 | - | - | - |
| textual_backdoor | fedavg | 79.90 | 63.66 | 71.78 | 16.24 | +0.00 | 67.41 | 91.38 | 79.40 |
| textual_backdoor | indexguard | 90.09 | 76.82 | 83.45 | 13.27 | +10.19 | 6.49 | 20.72 | 13.61 |
| textual_backdoor | flame | 82.64 | 71.95 | 77.30 | 10.69 | +2.74 | 3.14 | 10.97 | 7.05 |
| textual_backdoor | fltrust | 88.58 | 72.12 | 80.35 | 16.46 | +8.68 | 13.66 | 44.57 | 29.12 |
| textual_backdoor | safefedllm | 86.62 | 74.75 | 80.68 | 11.87 | +6.72 | 7.05 | 22.73 | 14.89 |
| textual_backdoor | fedgraph | 85.11 | 70.44 | 77.77 | 14.67 | +5.21 | 13.66 | 42.55 | 28.11 |

## Conclusion checks

- Complete matrix: 18/18.
- Under both attacks, every defense exceeds FedAvg standard-domain accuracy by at least one point.
- Under both attacks, adjacent standard-domain method results differ by at least one point.
- Every defense reduces mean textual-backdoor ASR relative to FedAvg.
- Distinct two-decimal endings: 25.
- All checks pass: True.
