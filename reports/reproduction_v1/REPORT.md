# FedSApro single-seed reproduction

Measured values are generated from raw run logs; paper targets remain separate columns.

Strict target-window acceptance: 3/18 cells.

Values are `measured (paper target)` in percentage points.

| Model | Attack | Method | HSSR | Standard acc. | Heterogeneous acc. | Accepted |
|---|---|---|---:|---:|---:|:---:|
| qwen2.5-1.5b | label_flip | fedavg | 0.00 (0.00) | 85.20 (83.50) | 78.30 (75.20) | no |
| qwen2.5-1.5b | label_flip | fedprox | 0.00 (30.20) | 84.95 (87.90) | 78.65 (76.60) | no |
| qwen2.5-1.5b | label_flip | flame | 8.33 (42.00) | 88.70 (89.70) | 84.30 (74.80) | no |
| qwen2.5-1.5b | label_flip | fltrust | 25.00 (45.00) | 88.90 (89.40) | 84.30 (73.20) | no |
| qwen2.5-1.5b | label_flip | indexguard | 0.00 (32.60) | 88.75 (88.90) | 84.45 (77.10) | no |
| qwen2.5-1.5b | label_flip | scaffold | 0.00 (27.70) | 88.05 (88.40) | 84.00 (77.60) | no |
| qwen2.5-1.5b | none | fedavg | 0.00 (0.00) | 88.80 (91.50) | 83.70 (84.60) | yes |
| qwen2.5-1.5b | none | fedprox | 0.00 (17.00) | 88.65 (91.40) | 83.60 (83.30) | no |
| qwen2.5-1.5b | none | flame | 33.33 (27.00) | 88.70 (91.40) | 83.95 (79.70) | no |
| qwen2.5-1.5b | none | fltrust | 16.67 (28.20) | 88.85 (91.10) | 84.05 (78.00) | no |
| qwen2.5-1.5b | none | indexguard | 16.67 (18.80) | 88.70 (91.20) | 83.75 (81.90) | yes |
| qwen2.5-1.5b | none | scaffold | 0.00 (14.50) | 88.00 (91.70) | 83.45 (84.10) | no |
| qwen2.5-1.5b | textual_backdoor | fedavg | 0.00 (0.00) | 88.65 (91.40) | 83.70 (82.70) | yes |
| qwen2.5-1.5b | textual_backdoor | fedprox | 0.00 (32.00) | 88.65 (90.60) | 83.95 (75.80) | no |
| qwen2.5-1.5b | textual_backdoor | flame | 0.00 (47.60) | 88.05 (90.30) | 83.35 (73.60) | no |
| qwen2.5-1.5b | textual_backdoor | fltrust | 16.67 (43.00) | 88.90 (90.50) | 84.10 (74.90) | no |
| qwen2.5-1.5b | textual_backdoor | indexguard | 0.00 (35.10) | 88.65 (90.80) | 83.70 (76.50) | no |
| qwen2.5-1.5b | textual_backdoor | scaffold | 0.00 (29.40) | 88.30 (90.90) | 83.40 (77.00) | no |

## Qualitative conclusion audit

6/9 robust-method comparisons show higher HSSR than the paired FedAvg baseline (mechanism-level support).

9/9 retain standard-domain accuracy within five percentage points of the paired FedAvg baseline.

0/9 show both higher HSSR and lower heterogeneous-domain accuracy (downstream-impact support).

0/9 robust-method comparisons satisfy all three predeclared checks: higher HSSR than FedAvg, lower heterogeneous-domain accuracy, and standard-domain accuracy retained within five points.
