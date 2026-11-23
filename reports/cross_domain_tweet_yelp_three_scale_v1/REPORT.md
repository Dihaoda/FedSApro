# TweetEval-Yelp three-scale validation

Measured seed-2026 binary-sentiment results for the frozen TweetEval standard domain and Yelp heterogeneous benign domain. Domain gap is the absolute accuracy disparity. HSSR, HRCW, paper targets, and synthetic jitter are not acceptance criteria.

## Scale overview

| Scale | Overall mean acc. | Clean mean | Label-flip mean | Backdoor mean | Clean FedAvg gap | Label FedAvg gap | Backdoor FedAvg gap | Min defense gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | 81.42 | 83.60 | 80.47 | 80.20 | 11.14 | 15.40 | 12.09 | 3.25 |
| 1.5B | 84.22 | 87.95 | 82.61 | 82.11 | 8.06 | 9.35 | 9.52 | 3.58 |
| 3B | 86.20 | 89.75 | 85.02 | 83.84 | 5.43 | 20.77 | 20.60 | 3.25 |

## Complete measured matrix

| Scale | Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Domain gap | ASR standard | ASR heterogeneous | ASR mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | none | fedavg | 90.31 | 79.17 | 84.74 | 11.14 | - | - | - |
| 0.5B | none | indexguard | 90.37 | 78.95 | 84.66 | 11.42 | - | - | - |
| 0.5B | none | flame | 89.14 | 78.56 | 83.85 | 10.58 | - | - | - |
| 0.5B | none | fltrust | 83.76 | 74.58 | 79.17 | 9.18 | - | - | - |
| 0.5B | none | safefedllm | 90.03 | 79.51 | 84.77 | 10.53 | - | - | - |
| 0.5B | none | fedgraph | 88.63 | 80.18 | 84.41 | 8.45 | - | - | - |
| 0.5B | label_flip | fedavg | 80.68 | 65.29 | 72.98 | 15.40 | - | - | - |
| 0.5B | label_flip | indexguard | 90.31 | 78.89 | 84.60 | 11.42 | - | - | - |
| 0.5B | label_flip | flame | 88.30 | 77.44 | 82.87 | 10.86 | - | - | - |
| 0.5B | label_flip | fltrust | 83.93 | 73.29 | 78.61 | 10.64 | - | - | - |
| 0.5B | label_flip | safefedllm | 84.99 | 77.04 | 81.02 | 7.95 | - | - | - |
| 0.5B | label_flip | fedgraph | 87.07 | 78.44 | 82.75 | 8.62 | - | - | - |
| 0.5B | textual_backdoor | fedavg | 78.44 | 66.35 | 72.40 | 12.09 | 37.51 | 65.96 | 51.74 |
| 0.5B | textual_backdoor | indexguard | 90.31 | 78.89 | 84.60 | 11.42 | 9.85 | 30.57 | 20.21 |
| 0.5B | textual_backdoor | flame | 88.30 | 77.44 | 82.87 | 10.86 | 13.55 | 31.02 | 22.28 |
| 0.5B | textual_backdoor | fltrust | 83.26 | 72.12 | 77.69 | 11.14 | 23.07 | 42.55 | 32.81 |
| 0.5B | textual_backdoor | safefedllm | 85.83 | 75.98 | 80.91 | 9.85 | 15.90 | 32.70 | 24.30 |
| 0.5B | textual_backdoor | fedgraph | 87.07 | 78.44 | 82.75 | 8.62 | 5.82 | 16.46 | 11.14 |
| 1.5B | none | fedavg | 92.27 | 84.21 | 88.24 | 8.06 | - | - | - |
| 1.5B | none | indexguard | 92.78 | 85.27 | 89.03 | 7.50 | - | - | - |
| 1.5B | none | flame | 90.15 | 82.14 | 86.14 | 8.01 | - | - | - |
| 1.5B | none | fltrust | 92.39 | 83.93 | 88.16 | 8.45 | - | - | - |
| 1.5B | none | safefedllm | 92.72 | 84.94 | 88.83 | 7.78 | - | - | - |
| 1.5B | none | fedgraph | 91.83 | 82.81 | 87.32 | 9.01 | - | - | - |
| 1.5B | label_flip | fedavg | 68.31 | 77.66 | 72.98 | 9.35 | - | - | - |
| 1.5B | label_flip | indexguard | 92.22 | 85.61 | 88.91 | 6.61 | - | - | - |
| 1.5B | label_flip | flame | 85.72 | 80.85 | 83.29 | 4.87 | - | - | - |
| 1.5B | label_flip | fltrust | 82.70 | 78.89 | 80.80 | 3.81 | - | - | - |
| 1.5B | label_flip | safefedllm | 88.41 | 79.40 | 83.90 | 9.01 | - | - | - |
| 1.5B | label_flip | fedgraph | 90.71 | 80.80 | 85.75 | 9.91 | - | - | - |
| 1.5B | textual_backdoor | fedavg | 78.33 | 68.81 | 73.57 | 9.52 | 44.90 | 63.83 | 54.37 |
| 1.5B | textual_backdoor | indexguard | 92.72 | 85.55 | 89.14 | 7.17 | 5.04 | 16.91 | 10.97 |
| 1.5B | textual_backdoor | flame | 84.27 | 78.44 | 81.35 | 5.82 | 26.76 | 42.44 | 34.60 |
| 1.5B | textual_backdoor | fltrust | 81.91 | 77.21 | 79.56 | 4.70 | 32.70 | 42.44 | 37.57 |
| 1.5B | textual_backdoor | safefedllm | 87.91 | 78.61 | 83.26 | 9.29 | 20.16 | 39.64 | 29.90 |
| 1.5B | textual_backdoor | fedgraph | 90.71 | 80.80 | 85.75 | 9.91 | 14.45 | 31.35 | 22.90 |
| 3B | none | fedavg | 92.39 | 86.95 | 89.67 | 5.43 | - | - | - |
| 3B | none | indexguard | 92.83 | 86.90 | 89.87 | 5.94 | - | - | - |
| 3B | none | flame | 92.72 | 86.34 | 89.53 | 6.38 | - | - | - |
| 3B | none | fltrust | 92.78 | 86.90 | 89.84 | 5.88 | - | - | - |
| 3B | none | safefedllm | 92.39 | 86.90 | 89.64 | 5.49 | - | - | - |
| 3B | none | fedgraph | 92.67 | 87.23 | 89.95 | 5.43 | - | - | - |
| 3B | label_flip | fedavg | 83.31 | 62.54 | 72.93 | 20.77 | - | - | - |
| 3B | label_flip | indexguard | 92.67 | 86.67 | 89.67 | 5.99 | - | - | - |
| 3B | label_flip | flame | 90.82 | 86.06 | 88.44 | 4.76 | - | - | - |
| 3B | label_flip | fltrust | 89.75 | 84.21 | 86.98 | 5.54 | - | - | - |
| 3B | label_flip | safefedllm | 86.56 | 84.04 | 85.30 | 2.52 | - | - | - |
| 3B | label_flip | fedgraph | 88.07 | 85.50 | 86.79 | 2.58 | - | - | - |
| 3B | textual_backdoor | fedavg | 78.67 | 58.06 | 68.37 | 20.60 | 55.88 | 93.73 | 74.80 |
| 3B | textual_backdoor | indexguard | 92.67 | 86.67 | 89.67 | 5.99 | 3.36 | 21.84 | 12.60 |
| 3B | textual_backdoor | flame | 89.14 | 81.41 | 85.27 | 7.73 | 16.46 | 39.98 | 28.22 |
| 3B | textual_backdoor | fltrust | 86.90 | 81.58 | 84.24 | 5.32 | 26.09 | 39.19 | 32.64 |
| 3B | textual_backdoor | safefedllm | 91.04 | 86.34 | 88.69 | 4.70 | 12.43 | 18.92 | 15.68 |
| 3B | textual_backdoor | fedgraph | 88.07 | 85.50 | 86.79 | 2.58 | 21.61 | 20.27 | 20.94 |

Each scale's machine-readable conclusion checks and compact raw evidence are stored in its dedicated report directory.
