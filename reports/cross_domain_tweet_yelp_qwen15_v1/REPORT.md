# TweetEval-Yelp Qwen2.5-1.5B validation study

Measured seed-2026 binary-sentiment results on the frozen TweetEval-Yelp partition. HSSR, HRCW, paper targets, and synthetic jitter are not used.

| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Mean delta vs 0.5B | Domain gap | ASR standard | ASR heterogeneous | ASR mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | fedavg | 92.27 | 84.21 | 88.24 | +3.50 | 8.06 | - | - | - |
| none | indexguard | 92.78 | 85.27 | 89.03 | +4.37 | 7.50 | - | - | - |
| none | flame | 90.15 | 82.14 | 86.14 | +2.30 | 8.01 | - | - | - |
| none | fltrust | 92.39 | 83.93 | 88.16 | +8.99 | 8.45 | - | - | - |
| none | safefedllm | 92.72 | 84.94 | 88.83 | +4.06 | 7.78 | - | - | - |
| none | fedgraph | 91.83 | 82.81 | 87.32 | +2.91 | 9.01 | - | - | - |
| label_flip | fedavg | 68.31 | 77.66 | 72.98 | +0.00 | 9.35 | - | - | - |
| label_flip | indexguard | 92.22 | 85.61 | 88.91 | +4.31 | 6.61 | - | - | - |
| label_flip | flame | 85.72 | 80.85 | 83.29 | +0.42 | 4.87 | - | - | - |
| label_flip | fltrust | 82.70 | 78.89 | 80.80 | +2.18 | 3.81 | - | - | - |
| label_flip | safefedllm | 88.41 | 79.40 | 83.90 | +2.88 | 9.01 | - | - | - |
| label_flip | fedgraph | 90.71 | 80.80 | 85.75 | +3.00 | 9.91 | - | - | - |
| textual_backdoor | fedavg | 78.33 | 68.81 | 73.57 | +1.18 | 9.52 | 44.90 | 63.83 | 54.37 |
| textual_backdoor | indexguard | 92.72 | 85.55 | 89.14 | +4.54 | 7.17 | 5.04 | 16.91 | 10.97 |
| textual_backdoor | flame | 84.27 | 78.44 | 81.35 | -1.51 | 5.82 | 26.76 | 42.44 | 34.60 |
| textual_backdoor | fltrust | 81.91 | 77.21 | 79.56 | +1.88 | 4.70 | 32.70 | 42.44 | 37.57 |
| textual_backdoor | safefedllm | 87.91 | 78.61 | 83.26 | +2.35 | 9.29 | 20.16 | 39.64 | 29.90 |
| textual_backdoor | fedgraph | 90.71 | 80.80 | 85.75 | +3.00 | 9.91 | 14.45 | 31.35 | 22.90 |

## Frozen conclusion checks

- Complete matrix: 18/18.
- Attacks increasing the FedAvg domain gap: 2/2.
- Every attacked defense exceeds FedAvg standard-domain accuracy by at least three points.
- Minimum observed attacked-defense standard-domain gain: 3.58 points.
- Adjacent attacked standard-domain results differ by at least one point.
- Every defense reduces mean textual-backdoor ASR relative to FedAvg.
- Improved cells over 0.5B: 16/18.
- Overall mean gain over 0.5B: +2.80 points.
- Every attack-condition mean improves over 0.5B: True.
- All checks pass: True.
