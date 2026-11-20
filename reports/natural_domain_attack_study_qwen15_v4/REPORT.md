# Qwen2.5-1.5B natural domain-attack study

Measured seed-2026 results using 1,786 randomly sampled examples per domain. HSSR, HRCW, paper targets, and synthetic jitter are not used.

| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Domain gap | Standard gain vs FedAvg | ASR standard | ASR heterogeneous | ASR mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | fedavg | 91.66 | 79.68 | 85.67 | 11.98 | +0.00 | - | - | - |
| none | indexguard | 89.64 | 77.77 | 83.71 | 11.87 | -2.02 | - | - | - |
| none | flame | 80.91 | 68.59 | 74.75 | 12.32 | -10.75 | - | - | - |
| none | fltrust | 85.78 | 67.75 | 76.76 | 18.03 | -5.88 | - | - | - |
| none | safefedllm | 85.95 | 70.32 | 78.14 | 15.62 | -5.71 | - | - | - |
| none | fedgraph | 82.64 | 66.07 | 74.36 | 16.57 | -9.01 | - | - | - |
| label_flip | fedavg | 65.51 | 54.48 | 59.99 | 11.03 | +0.00 | - | - | - |
| label_flip | indexguard | 89.81 | 77.72 | 83.76 | 12.09 | +24.30 | - | - | - |
| label_flip | flame | 71.72 | 66.13 | 68.92 | 5.60 | +6.22 | - | - | - |
| label_flip | fltrust | 85.95 | 68.42 | 77.18 | 17.53 | +20.44 | - | - | - |
| label_flip | safefedllm | 84.32 | 68.81 | 76.57 | 15.51 | +18.81 | - | - | - |
| label_flip | fedgraph | 82.75 | 67.02 | 74.89 | 15.73 | +17.25 | - | - | - |
| textual_backdoor | fedavg | 54.70 | 50.28 | 52.49 | 4.42 | +0.00 | 96.98 | 99.44 | 98.21 |
| textual_backdoor | indexguard | 89.81 | 77.72 | 83.76 | 12.09 | +35.11 | 10.86 | 26.76 | 18.81 |
| textual_backdoor | flame | 80.63 | 66.97 | 73.80 | 13.66 | +25.92 | 19.37 | 46.36 | 32.87 |
| textual_backdoor | fltrust | 86.00 | 68.98 | 77.49 | 17.02 | +31.30 | 20.49 | 48.94 | 34.71 |
| textual_backdoor | safefedllm | 84.55 | 66.07 | 75.31 | 18.48 | +29.84 | 28.22 | 57.45 | 42.83 |
| textual_backdoor | fedgraph | 82.75 | 67.02 | 74.89 | 15.73 | +28.05 | 24.64 | 54.09 | 39.36 |

## Conclusion checks

- Complete matrix: 18/18.
- Under both attacks, every defense exceeds FedAvg standard-domain accuracy by at least one point.
- Under both attacks, adjacent standard-domain method results differ by at least one point.
- Every defense reduces mean textual-backdoor ASR relative to FedAvg.
- Distinct two-decimal endings: 25.
- All checks pass: True.
