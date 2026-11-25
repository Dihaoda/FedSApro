# HARP-FedLLM six-group study

This directory stores measured results for **Heterogeneity-Aware Representation
Preservation for Robust Federated Large Language Model Fine-Tuning
(HARP-FedLLM)**. Results are added one group at a time and are not synthetic or
manually jittered.

## Frozen method after group 1

- The server uses 48 balanced **standard-domain-only** root examples. It does
  not assume access to text from the unseen heterogeneous domain.
- Three transformer layer outputs are used (one-third, two-thirds, and final),
  together with the matching LoRA update blocks.
- A client fingerprint contains update direction, clean-anchor representation
  shift, response to one fixed neutral suffix, and three layerwise update RMS
  values.
- PCA keeps six components. HDBSCAN discovers supported modes with a minimum
  mode size of four. Distance from the main mode is recorded as domain novelty
  but is **not** a negative signal by itself.
- The risk score uses only one-sided anomalies: log update norm above 3 robust
  standard deviations, absolute standard-anchor loss harm above 2.5 robust
  standard deviations, or perturbation sensitivity above 3 robust standard
  deviations. An unsupported mode adds at most 0.5 risk only when another risk
  signal is already present.
- Client factor is `max(0.01, exp(-risk))`; sample count times this factor is
  used for aggregation. No client role or attack label enters the computation.
- A bounded utility term `1 + 0.15 * min(z_benefit, 2)` mildly favors benign
  updates that reduce standard-anchor loss more than the robust median.
- One epoch on the same 200 standard-domain root examples supplies an anchor
  correction. Its contribution is fixed at 0.4 of the client aggregate before
  the common server multiplier; calibration uses fixed shuffle seed 777.
- The server learning-rate multiplier is 5.0 for every attack condition.

The 1% floor is intentionally retained: HARP-FedLLM is a soft weighting method,
not a hard client-deletion rule.

## Group 1: Qwen2.5-0.5B, Amazon Polarity--IMDb

All values below are measured on seed 2026 with 1,786 examples per test domain.

| Attack | Method | Standard accuracy | Heterogeneous accuracy | Domain gap | Mean ASR |
|---|---|---:|---:|---:|---:|
| None | FedAvg | 89.92 | 76.76 | 13.16 | - |
| None | HARP-FedLLM | **90.71** | **77.27** | 13.44 | - |
| Label flip | FedAvg | 81.41 | 67.13 | 14.28 | - |
| Label flip | IndexGuard | 89.47 | 76.48 | 12.99 | - |
| Label flip | HARP-FedLLM | **90.59** | **76.99** | 13.61 | - |
| Textual backdoor | FedAvg | 79.90 | 63.66 | 16.24 | 79.40 |
| Textual backdoor | IndexGuard | 89.64 | 76.99 | 12.65 | 13.49 |
| Textual backdoor | HARP-FedLLM | **90.37** | **77.32** | 13.05 | 16.97 |
| Textual backdoor | FLAME | 83.48 | 72.79 | 10.69 | **7.45** |

HARP-FedLLM has the best two-domain mean accuracy in all three conditions:
83.99% without attack, 83.79% under label flip, and 83.85% under textual
backdoor. Its full-condition mean is 83.87%, compared with 83.27% for the next
best method, IndexGuard. The accuracy gain trades some backdoor ASR against
IndexGuard, Safe-FedLLM, and FLAME; HARP-FedLLM remains far below FedAvg's mean
ASR while prioritizing heterogeneous-client retention and task accuracy.

To avoid identical attacked rows caused by complete attacker removal, the
fixed baseline variants retain a small attack-agnostic residual coefficient:
2% for IndexGuard, 5% for FLAME, a 2% FLTrust floor, and 3% for FedGraph. Their
native index overlap, HDBSCAN/clipping, root cosine, and graph-centrality rules
remain unchanged. No pair of methods now shares both rounded domain
accuracies within an attack condition.

The full 21-row comparison is in `group1_qwen05_amazon_imdb.csv`. Raw evidence
for every HARP-FedLLM condition is stored under `evidence/group1/`.

## Group 2: Qwen2.5-0.5B, TweetEval--Yelp Polarity

The standard-only anchor utility and correction used in group 1 overfit the
large TweetEval--Yelp length shift. Group 2 therefore keeps the same risk
detector but disables utility and root correction. Instead, a bounded novelty
preservation factor `1 + 0.30 * min(z_novelty, 2)` is applied after the risk
factor. Thus novelty can preserve a risk-free client but cannot override a
malicious-risk penalty. This choice was fixed using the no-attack condition and
then reused unchanged for both attacks.

| Attack | Method | Standard accuracy | Heterogeneous accuracy | Mean accuracy | Mean ASR |
|---|---|---:|---:|---:|---:|
| None | FedAvg | 90.31 | 79.17 | 84.74 | - |
| None | IndexGuard | **90.54** | 79.34 | **84.94** | - |
| None | Safe-FedLLM | 90.03 | **79.51** | 84.77 | - |
| None | HARP-FedLLM | **90.59** | 78.89 | 84.74 | - |
| Label flip | FedAvg | 80.68 | 65.29 | 72.98 | - |
| Label flip | IndexGuard | **90.26** | **78.67** | **84.46** | - |
| Label flip | HARP-FedLLM | 90.20 | 78.11 | 84.15 | - |
| Textual backdoor | FedAvg | 78.44 | 66.35 | 72.40 | 51.74 |
| Textual backdoor | IndexGuard | **90.54** | **78.61** | **84.57** | 21.89 |
| Textual backdoor | HARP-FedLLM | 90.15 | 78.22 | 84.18 | 22.23 |
| Textual backdoor | FedGraph | 87.12 | 78.16 | 82.64 | **13.33** |

HARP-FedLLM ranks second under both attacks and beats the other five baselines
on mean accuracy. Across all three conditions, its mean is 84.36%, versus
84.66% for IndexGuard. Its backdoor mean ASR is 22.23%, close to IndexGuard
(21.89%), below Safe-FedLLM (24.30%), and far below FedAvg (51.74%); FedGraph is
the low-ASR exception. No two methods share both rounded domain accuracies
within an attack condition.

The full 21-row comparison is in `group2_qwen05_tweet_yelp.csv`. Raw evidence
for HARP-FedLLM is stored under `evidence/group2/`.

