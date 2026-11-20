# Calibration and rejected runs

This trail separates calibration from the formal seed-2026 matrix. Rejected measurements are not
presented as final results. Their server-side raw directories were retained until the final GitHub
evidence bundle was produced; the compact record below is the durable hand-off.

## Domain-stress and attack-strength pilots

| Pilot | Standard | Heterogeneous | Gap | Decision |
|---|---:|---:|---:|---|
| `length24_v1`, clean | 75.70 | 71.80 | 3.90 | Reject: domain gap too small. |
| `length_extreme_v2`, clean | 91.80 | 69.95 | 21.85 | Clean gap adequate. |
| `length_extreme_v2`, label flip | 92.30 | 70.35 | 21.95 | Reject: attack increased the gap by only 0.10. |
| `imdb_attack_length_control_v3`, clean | 91.50 | 71.10 | 20.40 | Candidate baseline. |
| `imdb_attack_length_control_v3`, label flip | 79.30 | 53.65 | 25.65 | Reject: attacked gap increase only 5.25. |
| `lexical_control_v4`, clean | 89.00 | 71.10 | 17.90 | Reject: incomplete attack pair and weaker control. |
| label scale 3 | 75.00 | 50.30 | 24.70 | Reject: attacked gap below 30. |
| label scale 2 | 81.90 | 52.95 | 28.95 | Reject: attacked gap below 30. |

Scale-5 and scale-15 partial label pilots were interrupted before a valid summary and remain
explicitly non-results. The frozen label-flip choice is LoRA-only scale 5 under the final
`length_extremes` split; it yields `88.25 / 56.80` and a 31.45-point gap.

| Text-backdoor scale | Standard | Heterogeneous | Gap | ASR standard | ASR heterogeneous | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 93.15 | 75.95 | 17.20 | 9.60 | 32.50 | Reject: attack too weak. |
| 2 | 92.40 | 71.20 | 21.20 | 14.60 | 65.20 | Reject: gap increase too small. |
| 5 | 52.40 | 50.00 | 2.40 | 100.00 | 100.00 | Reject: utility collapse. |
| **3 (frozen)** | **89.90** | **54.60** | **35.30** | **34.10** | **99.40** | Formal stress setting. |

## Safe-FedLLM probe

Probe fitting uses only seed 9101, not formal seed 2026. It consumes 90 saved terminal-delta
records across clean, label-flip, and textual-backdoor calibration runs; six records are harmful.
The final linear probe's calibration accuracy, precision, recall, F1, and ROC AUC are all 1.0
(AUC differs from 1 only by floating-point rounding).

The first direction-only feature policy passed in-sample calibration but failed to generalize in a
formal diagnostic run (`87.55 / 54.70`, gap 32.85). That result was rejected and preserved. The
final feature policy appends `log1p(norm)` to the normalized first-layer LoRA-B direction. The new
feature and probe were fitted only from the existing seed-9101 calibration artifacts; formal
attacker identities were not used for fitting or threshold selection. The corrected formal result
is `92.10 / 70.40`, gap 21.70.

## FedGraph

Full-delta graphs failed to isolate the attack cluster at both tested settings:

| Variant | Standard | Heterogeneous | Gap | Decision |
|---|---:|---:|---:|---|
| full delta, threshold 0.80 | 86.60 | 53.50 | 33.10 | Reject: attackers retained. |
| full delta, threshold 0.50 | 83.95 | 51.40 | 32.55 | Reject: benign clients removed. |

A seed-9101 sweep over the first-layer LoRA-B calibration features selected threshold 0.50. It
separated all three malicious clients with zero benign false positives for both attacks. One
subsequent run stopped before aggregation because the layout argument was not forwarded; this
runtime failure produced no metric and was preserved as `fedgraph_first_lora_b_missing_layout`.

An interim KMeans implementation produced good metrics but was superseded after method review
confirmed that the FedGraph paper specifies HDBSCAN. Those three KMeans cells were moved to
`rejected_fedgraph_kmeans_v1` and excluded from the final table. The final implementation uses
`HDBSCAN(min_cluster_size=16, min_samples=1, allow_single_cluster=True)` and was rerun for all
three conditions.

## Frozen decision rule

Before the final export, the audit required:

- clean standard accuracy at least 85 and clean gap at least 15;
- each attacked FedAvg gap at least 10 points above clean;
- every attacked defense to gain at least 5 mean-accuracy points and 10 heterogeneous-accuracy
  points over attacked FedAvg while reducing the gap by at least 8 points;
- every textual-backdoor defense to reduce mean ASR by at least 30 points;
- clean mean-accuracy loss for every defense no greater than 2 points.

The final 18-cell matrix passes every rule. Exact floating-point values and per-cell decisions are
in `conclusion_checks.json`.
