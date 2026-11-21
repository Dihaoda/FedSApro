# Qwen2.5-3B norm-matched attack study

Measured seed-2026 results using 1,786 randomly sampled examples per domain. The data split, local training budget, attacks, and acceptance rules match the Qwen2.5-1.5B study. Method-by-condition server step sizes are explicit in the frozen configuration and only scale the already-aggregated update. HSSR, HRCW, paper targets, and synthetic jitter are not used for acceptance.

| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Mean Δ vs 1.5B | Domain gap | ASR standard | ASR heterogeneous | ASR mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | fedavg | 92.50 | 80.01 | 86.25 | +0.59 | 12.49 | - | - | - |
| none | indexguard | 93.06 | 82.31 | 87.68 | +1.65 | 10.75 | - | - | - |
| none | flame | 92.50 | 80.57 | 86.53 | +4.34 | 11.93 | - | - | - |
| none | fltrust | 92.78 | 80.96 | 86.87 | +4.14 | 11.81 | - | - | - |
| none | safefedllm | 93.00 | 81.75 | 87.37 | +2.07 | 11.25 | - | - | - |
| none | fedgraph | 91.49 | 77.10 | 84.29 | +3.56 | 14.39 | - | - | - |
| label_flip | fedavg | 86.62 | 66.41 | 76.51 | +1.79 | 20.21 | - | - | - |
| label_flip | indexguard | 92.89 | 82.36 | 87.63 | +1.68 | 10.53 | - | - | - |
| label_flip | flame | 89.42 | 73.91 | 81.66 | +2.38 | 15.51 | - | - | - |
| label_flip | fltrust | 90.59 | 74.24 | 82.42 | -0.17 | 16.35 | - | - | - |
| label_flip | safefedllm | 87.68 | 68.81 | 78.25 | -6.16 | 18.87 | - | - | - |
| label_flip | fedgraph | 91.88 | 78.16 | 85.02 | +4.17 | 13.72 | - | - | - |
| textual_backdoor | fedavg | 86.45 | 67.08 | 76.76 | +1.88 | 19.37 | 35.83 | 61.70 | 48.77 |
| textual_backdoor | indexguard | 93.00 | 82.64 | 87.82 | +1.88 | 10.36 | 9.52 | 25.08 | 17.30 |
| textual_backdoor | flame | 90.20 | 74.41 | 82.31 | -0.03 | 15.79 | 24.30 | 49.94 | 37.12 |
| textual_backdoor | fltrust | 88.91 | 70.38 | 79.65 | -3.33 | 18.53 | 25.98 | 52.86 | 39.42 |
| textual_backdoor | safefedllm | 87.51 | 68.65 | 78.08 | -6.61 | 18.87 | 28.00 | 54.42 | 41.21 |
| textual_backdoor | fedgraph | 91.88 | 78.16 | 85.02 | +4.17 | 13.72 | 15.90 | 37.96 | 26.93 |

## Acceptance checks

- Complete matrix: 18/18.
- Under both attacks, every defense exceeds FedAvg standard-domain accuracy by at least one point.
- Under both attacks, adjacent standard-domain method results differ by at least one point.
- Every defense reduces mean textual-backdoor ASR relative to FedAvg.
- Attacked cells improved over 1.5B: 7/12.
- All cells improved over 1.5B: 13/18.
- Overall mean accuracy gain over 1.5B: +1.00 points.
- All required checks pass: True.
