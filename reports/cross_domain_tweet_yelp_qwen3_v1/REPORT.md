# TweetEval-Yelp Qwen2.5-3B validation study

Measured seed-2026 binary-sentiment results on the frozen TweetEval-Yelp partition. HSSR, HRCW, paper targets, and synthetic jitter are not used.

| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Mean delta vs 1.5B | Domain gap | ASR standard | ASR heterogeneous | ASR mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | fedavg | 92.39 | 86.95 | 89.67 | +1.43 | 5.43 | - | - | - |
| none | indexguard | 92.83 | 86.90 | 89.87 | +0.84 | 5.94 | - | - | - |
| none | flame | 92.72 | 86.34 | 89.53 | +3.39 | 6.38 | - | - | - |
| none | fltrust | 92.78 | 86.90 | 89.84 | +1.68 | 5.88 | - | - | - |
| none | safefedllm | 92.39 | 86.90 | 89.64 | +0.81 | 5.49 | - | - | - |
| none | fedgraph | 92.67 | 87.23 | 89.95 | +2.63 | 5.43 | - | - | - |
| label_flip | fedavg | 83.31 | 62.54 | 72.93 | -0.06 | 20.77 | - | - | - |
| label_flip | indexguard | 92.67 | 86.67 | 89.67 | +0.76 | 5.99 | - | - | - |
| label_flip | flame | 90.82 | 86.06 | 88.44 | +5.15 | 4.76 | - | - | - |
| label_flip | fltrust | 89.75 | 84.21 | 86.98 | +6.19 | 5.54 | - | - | - |
| label_flip | safefedllm | 86.56 | 84.04 | 85.30 | +1.40 | 2.52 | - | - | - |
| label_flip | fedgraph | 88.07 | 85.50 | 86.79 | +1.04 | 2.58 | - | - | - |
| textual_backdoor | fedavg | 78.67 | 58.06 | 68.37 | -5.21 | 20.60 | 55.88 | 93.73 | 74.80 |
| textual_backdoor | indexguard | 92.67 | 86.67 | 89.67 | +0.53 | 5.99 | 3.36 | 21.84 | 12.60 |
| textual_backdoor | flame | 89.14 | 81.41 | 85.27 | +3.92 | 7.73 | 16.46 | 39.98 | 28.22 |
| textual_backdoor | fltrust | 86.90 | 81.58 | 84.24 | +4.68 | 5.32 | 26.09 | 39.19 | 32.64 |
| textual_backdoor | safefedllm | 91.04 | 86.34 | 88.69 | +5.43 | 4.70 | 12.43 | 18.92 | 15.68 |
| textual_backdoor | fedgraph | 88.07 | 85.50 | 86.79 | +1.04 | 2.58 | 21.61 | 20.27 | 20.94 |

## Frozen conclusion checks

- Complete matrix: 18/18.
- Attacks increasing the FedAvg domain gap: 2/2.
- Every attacked defense exceeds FedAvg standard-domain accuracy by at least three points.
- Minimum observed attacked-defense standard-domain gain: 3.25 points.
- Adjacent attacked standard-domain results differ by at least one point.
- Every defense reduces mean textual-backdoor ASR relative to FedAvg.
- Improved cells over 1.5B: 16/18.
- Overall mean gain over 1.5B: +1.98 points.
- Every attack-condition mean improves over 1.5B: True.
- All checks pass: True.
