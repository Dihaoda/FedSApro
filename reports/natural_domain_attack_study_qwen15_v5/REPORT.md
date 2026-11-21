# Qwen2.5-1.5B norm-matched attack study

Measured seed-2026 results using 1,786 randomly sampled examples per domain. Attacker updates are normalized against the benign median update norm. HSSR, HRCW, paper targets, and synthetic jitter are not used.

| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Mean Δ vs 0.5B | Domain gap | ASR standard | ASR heterogeneous | ASR mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | fedavg | 91.66 | 79.68 | 85.67 | +2.32 | 11.98 | - | - | - |
| none | indexguard | 91.88 | 80.18 | 86.03 | +2.49 | 11.70 | - | - | - |
| none | flame | 88.02 | 76.37 | 82.19 | -1.76 | 11.65 | - | - | - |
| none | fltrust | 89.64 | 75.81 | 82.73 | +2.13 | 13.83 | - | - | - |
| none | safefedllm | 91.27 | 79.34 | 85.30 | +3.42 | 11.93 | - | - | - |
| none | fedgraph | 87.18 | 74.30 | 80.74 | +1.99 | 12.88 | - | - | - |
| label_flip | fedavg | 82.70 | 66.74 | 74.72 | +0.45 | 15.96 | - | - | - |
| label_flip | indexguard | 91.94 | 79.96 | 85.95 | +2.49 | 11.98 | - | - | - |
| label_flip | flame | 85.67 | 72.90 | 79.28 | +1.99 | 12.77 | - | - | - |
| label_flip | fltrust | 88.97 | 76.20 | 82.59 | +2.24 | 12.77 | - | - | - |
| label_flip | safefedllm | 90.37 | 78.44 | 84.41 | +5.74 | 11.93 | - | - | - |
| label_flip | fedgraph | 87.01 | 74.69 | 80.85 | +3.08 | 12.32 | - | - | - |
| textual_backdoor | fedavg | 82.70 | 67.08 | 74.89 | +3.11 | 15.62 | 43.45 | 68.31 | 55.88 |
| textual_backdoor | indexguard | 91.94 | 79.96 | 85.95 | +2.49 | 11.98 | 6.05 | 17.47 | 11.76 |
| textual_backdoor | flame | 88.35 | 76.32 | 82.33 | +5.04 | 12.04 | 9.52 | 25.20 | 17.36 |
| textual_backdoor | fltrust | 89.42 | 76.54 | 82.98 | +2.63 | 12.88 | 12.43 | 31.13 | 21.78 |
| textual_backdoor | safefedllm | 90.71 | 78.67 | 84.69 | +4.00 | 12.04 | 10.97 | 24.97 | 17.97 |
| textual_backdoor | fedgraph | 87.01 | 74.69 | 80.85 | +3.08 | 12.32 | 12.54 | 34.60 | 23.57 |

## Acceptance checks

- Complete matrix: 18/18.
- Under both attacks, every defense exceeds FedAvg standard-domain accuracy by at least one point.
- Under both attacks, adjacent standard-domain method results differ by at least one point.
- Every defense reduces mean textual-backdoor ASR relative to FedAvg.
- Attacked cells improved over 0.5B: 12/12.
- Overall mean accuracy gain over 0.5B: +2.61 points.
- All checks pass: True.
