# Qwen2.5-0.5B heterogeneous benign-client suppression study

These are measured seed-2026 results. Paper target values, HSSR, and HRCW are not used.

| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Domain gap | FedAvg heterogeneous advantage | ASR standard | ASR heterogeneous | ASR mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | fedavg | 93.39 | 75.93 | 84.66 | 17.47 | -0.00 | - | - | - |
| none | indexguard | 93.09 | 71.82 | 82.46 | 21.27 | +4.10 | - | - | - |
| none | flame | 93.29 | 72.17 | 82.73 | 21.12 | +3.75 | - | - | - |
| none | fltrust | 93.49 | 72.62 | 83.06 | 20.87 | +3.30 | - | - | - |
| none | safefedllm | 93.39 | 74.77 | 84.08 | 18.62 | +1.15 | - | - | - |
| none | fedgraph | 92.64 | 69.77 | 81.21 | 22.87 | +6.16 | - | - | - |
| label_flip | fedavg | 92.44 | 74.07 | 83.26 | 18.37 | -0.00 | - | - | - |
| label_flip | indexguard | 93.34 | 72.67 | 83.01 | 20.67 | +1.40 | - | - | - |
| label_flip | flame | 92.79 | 70.02 | 81.41 | 22.77 | +4.05 | - | - | - |
| label_flip | fltrust | 93.54 | 72.47 | 83.01 | 21.07 | +1.60 | - | - | - |
| label_flip | safefedllm | 88.59 | 60.21 | 74.40 | 28.38 | +13.86 | - | - | - |
| label_flip | fedgraph | 92.64 | 70.82 | 81.73 | 21.82 | +3.25 | - | - | - |
| textual_backdoor | fedavg | 92.69 | 73.42 | 83.06 | 19.27 | -0.00 | 11.81 | 50.45 | 31.13 |
| textual_backdoor | indexguard | 93.34 | 72.67 | 83.01 | 20.67 | +0.75 | 5.11 | 31.13 | 18.12 |
| textual_backdoor | flame | 92.79 | 70.02 | 81.41 | 22.77 | +3.40 | 3.30 | 10.21 | 6.76 |
| textual_backdoor | fltrust | 93.54 | 72.47 | 83.01 | 21.07 | +0.95 | 6.61 | 37.14 | 21.87 |
| textual_backdoor | safefedllm | 92.99 | 72.92 | 82.96 | 20.07 | +0.50 | 4.20 | 19.02 | 11.61 |
| textual_backdoor | fedgraph | 92.64 | 70.82 | 81.73 | 21.82 | +2.60 | 5.21 | 36.04 | 20.62 |

## Conclusion checks

- Complete matrix: 18/18.
- Under both attacks, FedAvg has higher heterogeneous-domain accuracy than every defense.
- Every defense reduces mean textual-backdoor ASR relative to FedAvg.
- Both attacks increase the FedAvg domain gap relative to the clean run.
- Distinct two-decimal accuracy pairs: 14/18.
- All conclusion checks pass: True.

Exact margins and decisions are in `conclusion_checks.json`.
