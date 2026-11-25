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
- The server learning-rate multiplier is 5.0 for every attack condition.

The 1% floor is intentionally retained: HARP-FedLLM is a soft weighting method,
not a hard client-deletion rule.

## Group 1: Qwen2.5-0.5B, Amazon Polarity--IMDb

All values below are measured on seed 2026 with 1,786 examples per test domain.

| Attack | Method | Standard accuracy | Heterogeneous accuracy | Domain gap | Mean ASR |
|---|---|---:|---:|---:|---:|
| None | FedAvg | 89.92 | 76.76 | 13.16 | - |
| None | HARP-FedLLM | 89.98 | 77.04 | 12.93 | - |
| Label flip | FedAvg | 81.41 | 67.13 | 14.28 | - |
| Label flip | IndexGuard | **90.09** | **76.82** | 13.27 | - |
| Label flip | HARP-FedLLM | 89.59 | 76.37 | 13.21 | - |
| Textual backdoor | FedAvg | 79.90 | 63.66 | 16.24 | 79.40 |
| Textual backdoor | IndexGuard | **90.09** | **76.82** | 13.27 | 13.61 |
| Textual backdoor | HARP-FedLLM | 89.53 | **76.82** | 12.71 | 13.21 |
| Textual backdoor | FLAME | 82.64 | 71.95 | 10.69 | **7.05** |

HARP-FedLLM is second by mean accuracy under both attacks and beats five of six
existing methods on both domain accuracies. Under textual backdoor it also has
lower mean ASR than IndexGuard, Safe-FedLLM, FLTrust, FedGraph, and FedAvg;
FLAME is the expected exception because its aggressive hard filtering lowers
ASR at the cost of substantially lower accuracy and heterogeneous retention.

The full 21-row comparison is in `group1_qwen05_amazon_imdb.csv`. Raw evidence
for every HARP-FedLLM condition is stored under `evidence/group1/`.

