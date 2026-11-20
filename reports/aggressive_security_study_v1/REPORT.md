# Aggressive Qwen2.5-0.5B security study

All values are measured single-seed results from this study. No paper target values, HSSR, or HRCW are used.

| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Gap | Mean Δ vs FedAvg | Heterogeneous Δ | Gap Δ | ASR standard | ASR heterogeneous | ASR mean | ASR Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | fedavg | 93.25 | 75.75 | 84.50 | 17.50 | +0.00 | +0.00 | +0.00 | - | - | - | - |
| none | indexguard | 93.35 | 75.90 | 84.62 | 17.45 | +0.12 | +0.15 | -0.05 | - | - | - | - |
| none | flame | 93.35 | 75.70 | 84.53 | 17.65 | +0.03 | -0.05 | +0.15 | - | - | - | - |
| none | fltrust | 93.65 | 75.90 | 84.78 | 17.75 | +0.28 | +0.15 | +0.25 | - | - | - | - |
| none | safefedllm | 93.35 | 75.95 | 84.65 | 17.40 | +0.15 | +0.20 | -0.10 | - | - | - | - |
| none | fedgraph | 93.00 | 75.25 | 84.12 | 17.75 | -0.38 | -0.50 | +0.25 | - | - | - | - |
| label_flip | fedavg | 88.25 | 56.80 | 72.53 | 31.45 | +0.00 | +0.00 | +0.00 | - | - | - | - |
| label_flip | indexguard | 93.20 | 75.65 | 84.42 | 17.55 | +11.90 | +18.85 | -13.90 | - | - | - | - |
| label_flip | flame | 93.15 | 75.75 | 84.45 | 17.40 | +11.92 | +18.95 | -14.05 | - | - | - | - |
| label_flip | fltrust | 93.60 | 76.30 | 84.95 | 17.30 | +12.42 | +19.50 | -14.15 | - | - | - | - |
| label_flip | safefedllm | 92.10 | 70.40 | 81.25 | 21.70 | +8.72 | +13.60 | -9.75 | - | - | - | - |
| label_flip | fedgraph | 93.10 | 75.55 | 84.33 | 17.55 | +11.80 | +18.75 | -13.90 | - | - | - | - |
| textual_backdoor | fedavg | 89.90 | 54.60 | 72.25 | 35.30 | +0.00 | +0.00 | +0.00 | 34.10 | 99.40 | 66.75 | +0.00 |
| textual_backdoor | indexguard | 93.20 | 75.65 | 84.42 | 17.55 | +12.17 | +21.05 | -17.75 | 6.70 | 19.30 | 13.00 | -53.75 |
| textual_backdoor | flame | 93.30 | 76.05 | 84.68 | 17.25 | +12.43 | +21.45 | -18.05 | 6.70 | 19.30 | 13.00 | -53.75 |
| textual_backdoor | fltrust | 93.60 | 76.30 | 84.95 | 17.30 | +12.70 | +21.70 | -18.00 | 6.00 | 21.00 | 13.50 | -53.25 |
| textual_backdoor | safefedllm | 92.90 | 74.35 | 83.62 | 18.55 | +11.38 | +19.75 | -16.75 | 11.40 | 44.50 | 27.95 | -38.80 |
| textual_backdoor | fedgraph | 93.10 | 75.55 | 84.33 | 17.55 | +12.08 | +20.95 | -17.75 | 5.80 | 13.90 | 9.85 | -56.90 |

## Predeclared aggressive checks

- Complete matrix: 18/18 cells.
- Attacked defense cells passing every utility/gap/ASR bar: 10/10.
- All predeclared checks pass: True.

Exact thresholds and per-cell decisions are in `conclusion_checks.json`.
