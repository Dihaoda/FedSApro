# Qwen2.5-0.5B accuracy-centric reproduction

This report uses only measured single-seed results from the present study. It does not compare against paper values and does not use HSSR or HRCW.

Accuracy, domain gap, and ASR are percentage points. `Mean acc. delta` is relative to FedAvg under the same attack. `Gap delta (clean)` is relative to the same method without an attack. Negative ASR delta is better.

| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Domain gap | Mean acc. delta | Gap delta (clean) | ASR standard | ASR heterogeneous | ASR delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | fedavg | 86.65 | 81.15 | 83.90 | 5.50 | +0.00 | +0.00 | - | - | - |
| none | fedprox | 86.30 | 81.80 | 84.05 | 4.50 | +0.15 | +0.00 | - | - | - |
| none | scaffold | 86.25 | 81.80 | 84.03 | 4.45 | +0.12 | +0.00 | - | - | - |
| none | indexguard | 86.65 | 81.15 | 83.90 | 5.50 | +0.00 | +0.00 | - | - | - |
| none | flame | 86.00 | 80.25 | 83.12 | 5.75 | -0.78 | +0.00 | - | - | - |
| none | fltrust | 86.95 | 81.15 | 84.05 | 5.80 | +0.15 | +0.00 | - | - | - |
| label_flip | fedavg | 85.10 | 79.00 | 82.05 | 6.10 | +0.00 | +0.60 | - | - | - |
| label_flip | fedprox | 86.05 | 80.45 | 83.25 | 5.60 | +1.20 | +1.10 | - | - | - |
| label_flip | scaffold | 86.05 | 80.15 | 83.10 | 5.90 | +1.05 | +1.45 | - | - | - |
| label_flip | indexguard | 86.70 | 81.35 | 84.03 | 5.35 | +1.98 | -0.15 | - | - | - |
| label_flip | flame | 86.70 | 80.60 | 83.65 | 6.10 | +1.60 | +0.35 | - | - | - |
| label_flip | fltrust | 86.90 | 81.00 | 83.95 | 5.90 | +1.90 | +0.10 | - | - | - |
| textual_backdoor | fedavg | 86.70 | 80.60 | 83.65 | 6.10 | +0.00 | +0.60 | 49.60 | 59.50 | +0.00 |
| textual_backdoor | fedprox | 86.55 | 81.80 | 84.18 | 4.75 | +0.53 | +0.25 | 41.60 | 48.20 | -9.65 |
| textual_backdoor | scaffold | 86.70 | 81.35 | 84.03 | 5.35 | +0.38 | +0.90 | 44.10 | 50.30 | -7.35 |
| textual_backdoor | indexguard | 86.90 | 81.10 | 84.00 | 5.80 | +0.35 | +0.30 | 17.20 | 22.60 | -34.65 |
| textual_backdoor | flame | 86.50 | 81.30 | 83.90 | 5.20 | +0.25 | -0.55 | 22.60 | 30.60 | -27.95 |
| textual_backdoor | fltrust | 87.25 | 81.40 | 84.32 | 5.85 | +0.67 | +0.05 | 14.20 | 17.50 | -38.70 |

## Predeclared conclusion checks

- Complete attack/method matrix: 18/18 cells.
- Attacks increasing the paired FedAvg domain gap: 2/2.
- Attacked defenses outperforming paired FedAvg on mean accuracy, and also on ASR for textual backdoor: 10/10.

The machine-readable decisions and exact deltas are in `conclusion_checks.json`.
