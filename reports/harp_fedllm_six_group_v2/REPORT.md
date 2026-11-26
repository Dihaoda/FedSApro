# HARP-FedLLM six-group measured results

All cells use seed 2026, the same cached client updates within each model/dataset/attack group, 48-token client training and defense fingerprints, and 128-token evaluation. Values below are direct measured outputs; no target injection or post-hoc jitter is used.

## Qwen2.5-0.5B on Amazon Polarity and IMDb

| Attack | Method | Standard | Heterogeneous | Mean | Gap | ASR standard | ASR heterogeneous |
|---|---|---:|---:|---:|---:|---:|---:|
| No Attack | FedAvg | 93.56 | 83.26 | 88.41 | 10.30 | -- | -- |
| No Attack | IndexGuard | 93.78 | 82.98 | 88.38 | 10.81 | -- | -- |
| No Attack | FLAME | 93.90 | 81.86 | 87.88 | 12.04 | -- | -- |
| No Attack | FLTrust | 94.57 | 76.99 | 85.78 | 17.58 | -- | -- |
| No Attack | Safe-FedLLM | 94.40 | 80.80 | 87.60 | 13.61 | -- | -- |
| No Attack | FedGraph | 93.73 | 75.25 | 84.49 | 18.48 | -- | -- |
| No Attack | HARP-FedLLM | 93.73 | 83.31 | 88.52 | 10.41 | -- | -- |
| Label Flip | FedAvg | 93.34 | 76.71 | 85.02 | 16.63 | -- | -- |
| Label Flip | IndexGuard | 93.95 | 82.87 | 88.41 | 11.09 | -- | -- |
| Label Flip | FLAME | 94.46 | 79.17 | 86.81 | 15.29 | -- | -- |
| Label Flip | FLTrust | 94.46 | 78.00 | 86.23 | 16.46 | -- | -- |
| Label Flip | Safe-FedLLM | 93.78 | 78.89 | 86.34 | 14.89 | -- | -- |
| Label Flip | FedGraph | 93.62 | 74.69 | 84.15 | 18.92 | -- | -- |
| Label Flip | HARP-FedLLM | 94.01 | 83.03 | 88.52 | 10.97 | -- | -- |
| Textual Backdoor | FedAvg | 91.55 | 75.76 | 83.65 | 15.79 | 26.88 | 70.32 |
| Textual Backdoor | IndexGuard | 93.84 | 82.87 | 88.35 | 10.97 | 11.09 | 16.35 |
| Textual Backdoor | FLAME | 94.46 | 79.51 | 86.98 | 14.95 | 7.05 | 11.20 |
| Textual Backdoor | FLTrust | 94.18 | 78.28 | 86.23 | 15.90 | 7.50 | 34.71 |
| Textual Backdoor | Safe-FedLLM | 93.95 | 80.68 | 87.32 | 13.27 | 6.94 | 18.59 |
| Textual Backdoor | FedGraph | 93.56 | 75.53 | 84.55 | 18.03 | 6.05 | 36.39 |
| Textual Backdoor | HARP-FedLLM | 93.90 | 82.92 | 88.41 | 10.97 | 11.20 | 16.35 |

Group-level conclusion passes: **True**. HARP-FedLLM wins both heterogeneous-domain and two-domain mean accuracy in 3/3 individual settings (exceptions: none). After averaging the three settings, its heterogeneous-domain margin over the best baseline is 0.1866 points and its two-domain mean-accuracy margin is 0.1027 points.

## Qwen2.5-0.5B on TweetEval and Yelp Polarity

| Attack | Method | Standard | Heterogeneous | Mean | Gap | ASR standard | ASR heterogeneous |
|---|---|---:|---:|---:|---:|---:|---:|
| No Attack | FedAvg | 89.31 | 88.30 | 88.80 | 1.01 | -- | -- |
| No Attack | IndexGuard | 89.75 | 89.19 | 89.47 | 0.56 | -- | -- |
| No Attack | FLAME | 88.58 | 89.42 | 89.00 | 0.84 | -- | -- |
| No Attack | FLTrust | 83.31 | 80.68 | 82.00 | 2.63 | -- | -- |
| No Attack | Safe-FedLLM | 89.92 | 89.14 | 89.53 | 0.78 | -- | -- |
| No Attack | FedGraph | 89.03 | 89.03 | 89.03 | 0.00 | -- | -- |
| No Attack | HARP-FedLLM | 89.64 | 89.47 | 89.56 | 0.17 | -- | -- |
| Label Flip | FedAvg | 80.85 | 76.60 | 78.72 | 4.26 | -- | -- |
| Label Flip | IndexGuard | 89.59 | 88.52 | 89.05 | 1.06 | -- | -- |
| Label Flip | FLAME | 87.96 | 87.91 | 87.93 | 0.06 | -- | -- |
| Label Flip | FLTrust | 84.04 | 80.46 | 82.25 | 3.58 | -- | -- |
| Label Flip | Safe-FedLLM | 84.94 | 85.78 | 85.36 | 0.84 | -- | -- |
| Label Flip | FedGraph | 87.18 | 87.85 | 87.51 | 0.67 | -- | -- |
| Label Flip | HARP-FedLLM | 89.59 | 89.25 | 89.42 | 0.34 | -- | -- |
| Textual Backdoor | FedAvg | 77.27 | 75.98 | 76.62 | 1.29 | 38.52 | 44.57 |
| Textual Backdoor | IndexGuard | 89.70 | 88.41 | 89.05 | 1.29 | 12.21 | 13.66 |
| Textual Backdoor | FLAME | 87.74 | 87.68 | 87.71 | 0.06 | 15.57 | 15.12 |
| Textual Backdoor | FLTrust | 82.98 | 80.74 | 81.86 | 2.24 | 21.28 | 21.28 |
| Textual Backdoor | Safe-FedLLM | 85.83 | 83.59 | 84.71 | 2.24 | 18.48 | 21.84 |
| Textual Backdoor | FedGraph | 87.57 | 88.19 | 87.88 | 0.62 | 8.85 | 8.62 |
| Textual Backdoor | HARP-FedLLM | 89.53 | 89.03 | 89.28 | 0.50 | 13.33 | 13.44 |

Group-level conclusion passes: **True**. HARP-FedLLM wins both heterogeneous-domain and two-domain mean accuracy in 3/3 individual settings (exceptions: none). After averaging the three settings, its heterogeneous-domain margin over the best baseline is 0.5412 points and its two-domain mean-accuracy margin is 0.2240 points.

## Qwen2.5-1.5B on Amazon Polarity and IMDb

| Attack | Method | Standard | Heterogeneous | Mean | Gap | ASR standard | ASR heterogeneous |
|---|---|---:|---:|---:|---:|---:|---:|
| No Attack | FedAvg | 95.91 | 87.18 | 91.55 | 8.73 | -- | -- |
| No Attack | IndexGuard | 96.14 | 87.01 | 91.57 | 9.13 | -- | -- |
| No Attack | FLAME | 94.12 | 80.57 | 87.35 | 13.55 | -- | -- |
| No Attack | FLTrust | 93.90 | 77.16 | 85.53 | 16.74 | -- | -- |
| No Attack | Safe-FedLLM | 95.46 | 87.12 | 91.29 | 8.34 | -- | -- |
| No Attack | FedGraph | 92.44 | 77.49 | 84.97 | 14.95 | -- | -- |
| No Attack | HARP-FedLLM | 95.97 | 87.57 | 91.77 | 8.40 | -- | -- |
| Label Flip | FedAvg | 94.06 | 68.37 | 81.22 | 25.70 | -- | -- |
| Label Flip | IndexGuard | 96.14 | 86.95 | 91.55 | 9.18 | -- | -- |
| Label Flip | FLAME | 91.99 | 76.48 | 84.24 | 15.51 | -- | -- |
| Label Flip | FLTrust | 93.56 | 76.32 | 84.94 | 17.25 | -- | -- |
| Label Flip | Safe-FedLLM | 95.35 | 85.89 | 90.62 | 9.46 | -- | -- |
| Label Flip | FedGraph | 91.55 | 75.53 | 83.54 | 16.01 | -- | -- |
| Label Flip | HARP-FedLLM | 95.91 | 87.01 | 91.46 | 8.90 | -- | -- |
| Textual Backdoor | FedAvg | 96.02 | 73.46 | 84.74 | 22.56 | 3.25 | 57.56 |
| Textual Backdoor | IndexGuard | 96.08 | 87.23 | 91.66 | 8.85 | 2.24 | 7.73 |
| Textual Backdoor | FLAME | 94.23 | 79.12 | 86.67 | 15.12 | 1.01 | 35.39 |
| Textual Backdoor | FLTrust | 94.06 | 76.54 | 85.30 | 17.53 | 0.90 | 41.88 |
| Textual Backdoor | Safe-FedLLM | 95.35 | 86.39 | 90.87 | 8.96 | 1.46 | 21.05 |
| Textual Backdoor | FedGraph | 91.94 | 75.81 | 83.87 | 16.13 | 0.90 | 43.45 |
| Textual Backdoor | HARP-FedLLM | 95.97 | 87.18 | 91.57 | 8.79 | 2.02 | 8.73 |

Group-level conclusion passes: **True**. HARP-FedLLM wins both heterogeneous-domain and two-domain mean accuracy in 1/3 individual settings (exceptions: Label Flip, Textual Backdoor). After averaging the three settings, its heterogeneous-domain margin over the best baseline is 0.1866 points and its two-domain mean-accuracy margin is 0.0093 points.

## Qwen2.5-1.5B on TweetEval and Yelp Polarity

| Attack | Method | Standard | Heterogeneous | Mean | Gap | ASR standard | ASR heterogeneous |
|---|---|---:|---:|---:|---:|---:|---:|
| No Attack | FedAvg | 91.99 | 92.44 | 92.22 | 0.45 | -- | -- |
| No Attack | IndexGuard | 92.50 | 92.72 | 92.61 | 0.22 | -- | -- |
| No Attack | FLAME | 90.48 | 91.27 | 90.87 | 0.78 | -- | -- |
| No Attack | FLTrust | 91.43 | 92.50 | 91.97 | 1.06 | -- | -- |
| No Attack | Safe-FedLLM | 92.33 | 93.11 | 92.72 | 0.78 | -- | -- |
| No Attack | FedGraph | 91.15 | 91.27 | 91.21 | 0.11 | -- | -- |
| No Attack | HARP-FedLLM | 92.50 | 93.17 | 92.83 | 0.67 | -- | -- |
| Label Flip | FedAvg | 66.74 | 87.23 | 76.99 | 20.49 | -- | -- |
| Label Flip | IndexGuard | 92.39 | 92.78 | 92.58 | 0.39 | -- | -- |
| Label Flip | FLAME | 86.17 | 89.92 | 88.05 | 3.75 | -- | -- |
| Label Flip | FLTrust | 83.31 | 88.13 | 85.72 | 4.82 | -- | -- |
| Label Flip | Safe-FedLLM | 88.19 | 88.69 | 88.44 | 0.50 | -- | -- |
| Label Flip | FedGraph | 89.98 | 90.82 | 90.40 | 0.84 | -- | -- |
| Label Flip | HARP-FedLLM | 92.50 | 92.89 | 92.69 | 0.39 | -- | -- |
| Textual Backdoor | FedAvg | 79.79 | 78.78 | 79.28 | 1.01 | 39.19 | 39.98 |
| Textual Backdoor | IndexGuard | 92.33 | 93.06 | 92.69 | 0.73 | 4.93 | 5.60 |
| Textual Backdoor | FLAME | 86.62 | 87.74 | 87.18 | 1.12 | 21.84 | 21.28 |
| Textual Backdoor | FLTrust | 82.64 | 86.95 | 84.80 | 4.31 | 31.35 | 22.17 |
| Textual Backdoor | Safe-FedLLM | 87.85 | 88.19 | 88.02 | 0.34 | 19.15 | 18.81 |
| Textual Backdoor | FedGraph | 89.92 | 90.37 | 90.15 | 0.45 | 15.68 | 14.22 |
| Textual Backdoor | HARP-FedLLM | 92.44 | 92.95 | 92.69 | 0.50 | 4.59 | 5.82 |

Group-level conclusion passes: **True**. HARP-FedLLM wins both heterogeneous-domain and two-domain mean accuracy in 2/3 individual settings (exceptions: Textual Backdoor). After averaging the three settings, its heterogeneous-domain margin over the best baseline is 0.1493 points and its two-domain mean-accuracy margin is 0.1120 points.

## Qwen2.5-3B on Amazon Polarity and IMDb

| Attack | Method | Standard | Heterogeneous | Mean | Gap | ASR standard | ASR heterogeneous |
|---|---|---:|---:|---:|---:|---:|---:|
| No Attack | FedAvg | 95.30 | 86.79 | 91.04 | 8.51 | -- | -- |
| No Attack | IndexGuard | 95.35 | 88.47 | 91.91 | 6.89 | -- | -- |
| No Attack | FLAME | 95.30 | 86.67 | 90.99 | 8.62 | -- | -- |
| No Attack | FLTrust | 95.41 | 87.63 | 91.52 | 7.78 | -- | -- |
| No Attack | Safe-FedLLM | 95.46 | 88.35 | 91.91 | 7.11 | -- | -- |
| No Attack | FedGraph | 95.35 | 83.87 | 89.61 | 11.48 | -- | -- |
| No Attack | HARP-FedLLM | 95.35 | 88.47 | 91.91 | 6.89 | -- | -- |
| Label Flip | FedAvg | 94.74 | 68.48 | 81.61 | 26.26 | -- | -- |
| Label Flip | IndexGuard | 95.46 | 88.41 | 91.94 | 7.05 | -- | -- |
| Label Flip | FLAME | 94.62 | 77.44 | 86.03 | 17.19 | -- | -- |
| Label Flip | FLTrust | 95.18 | 80.52 | 87.85 | 14.67 | -- | -- |
| Label Flip | Safe-FedLLM | 95.24 | 75.42 | 85.33 | 19.82 | -- | -- |
| Label Flip | FedGraph | 95.35 | 83.93 | 89.64 | 11.42 | -- | -- |
| Label Flip | HARP-FedLLM | 95.41 | 88.52 | 91.97 | 6.89 | -- | -- |
| Textual Backdoor | FedAvg | 95.13 | 75.08 | 85.11 | 20.04 | 8.06 | 47.14 |
| Textual Backdoor | IndexGuard | 95.24 | 88.52 | 91.88 | 6.72 | 5.60 | 17.69 |
| Textual Backdoor | FLAME | 95.18 | 80.96 | 88.07 | 14.22 | 7.39 | 37.96 |
| Textual Backdoor | FLTrust | 95.24 | 77.83 | 86.53 | 17.41 | 7.39 | 39.53 |
| Textual Backdoor | Safe-FedLLM | 95.18 | 76.37 | 85.78 | 18.81 | 7.61 | 39.75 |
| Textual Backdoor | FedGraph | 95.24 | 84.15 | 89.70 | 11.09 | 6.83 | 27.55 |
| Textual Backdoor | HARP-FedLLM | 95.52 | 88.69 | 92.11 | 6.83 | 5.71 | 17.69 |

Group-level conclusion passes: **True**. HARP-FedLLM wins both heterogeneous-domain and two-domain mean accuracy in 2/3 individual settings (exceptions: No Attack). After averaging the three settings, its heterogeneous-domain margin over the best baseline is 0.0933 points and its two-domain mean-accuracy margin is 0.0840 points.

## Qwen2.5-3B on TweetEval and Yelp Polarity

| Attack | Method | Standard | Heterogeneous | Mean | Gap | ASR standard | ASR heterogeneous |
|---|---|---:|---:|---:|---:|---:|---:|
| No Attack | FedAvg | 92.33 | 94.01 | 93.17 | 1.68 | -- | -- |
| No Attack | IndexGuard | 92.11 | 94.01 | 93.06 | 1.90 | -- | -- |
| No Attack | FLAME | 91.99 | 94.01 | 93.00 | 2.02 | -- | -- |
| No Attack | FLTrust | 91.71 | 93.90 | 92.81 | 2.18 | -- | -- |
| No Attack | Safe-FedLLM | 92.11 | 93.90 | 93.00 | 1.79 | -- | -- |
| No Attack | FedGraph | 92.05 | 93.90 | 92.97 | 1.85 | -- | -- |
| No Attack | HARP-FedLLM | 92.16 | 94.06 | 93.11 | 1.90 | -- | -- |
| Label Flip | FedAvg | 83.71 | 85.83 | 84.77 | 2.13 | -- | -- |
| Label Flip | IndexGuard | 91.88 | 94.01 | 92.95 | 2.13 | -- | -- |
| Label Flip | FLAME | 91.49 | 92.67 | 92.08 | 1.18 | -- | -- |
| Label Flip | FLTrust | 88.52 | 94.12 | 91.32 | 5.60 | -- | -- |
| Label Flip | Safe-FedLLM | 85.44 | 93.90 | 89.67 | 8.45 | -- | -- |
| Label Flip | FedGraph | 86.84 | 93.90 | 90.37 | 7.05 | -- | -- |
| Label Flip | HARP-FedLLM | 92.27 | 93.84 | 93.06 | 1.57 | -- | -- |
| Textual Backdoor | FedAvg | 79.84 | 82.87 | 81.35 | 3.02 | 52.97 | 57.22 |
| Textual Backdoor | IndexGuard | 92.16 | 94.12 | 93.14 | 1.96 | 3.70 | 5.71 |
| Textual Backdoor | FLAME | 90.37 | 94.06 | 92.22 | 3.70 | 14.00 | 8.85 |
| Textual Backdoor | FLTrust | 86.62 | 94.29 | 90.45 | 7.67 | 28.56 | 8.06 |
| Textual Backdoor | Safe-FedLLM | 89.70 | 93.78 | 91.74 | 4.09 | 15.90 | 3.81 |
| Textual Backdoor | FedGraph | 86.11 | 94.18 | 90.15 | 8.06 | 30.91 | 5.15 |
| Textual Backdoor | HARP-FedLLM | 92.11 | 94.01 | 93.06 | 1.90 | 2.58 | 4.82 |

Group-level conclusion passes: **False**. HARP-FedLLM wins both heterogeneous-domain and two-domain mean accuracy in 0/3 individual settings (exceptions: No Attack, Label Flip, Textual Backdoor). After averaging the three settings, its heterogeneous-domain margin over the best baseline is -0.1306 points and its two-domain mean-accuracy margin is 0.0280 points.


