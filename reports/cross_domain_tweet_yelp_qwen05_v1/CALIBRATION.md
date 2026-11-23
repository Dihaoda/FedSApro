# TweetEval-Yelp calibration record

This file records the bounded exploratory calibration that preceded the frozen
Qwen2.5-0.5B matrix. The final table is measured directly from model runs; no
synthetic jitter or manual modification of metrics is used.

## Dataset choice

TweetEval sentiment (neutral rows excluded) is the standard client domain and
Yelp Polarity is the heterogeneous benign domain. Both retain their source test
splits and are mapped to negative/positive labels. An earlier Yelp-standard /
TweetEval-heterogeneous orientation was rejected because clean FedAvg reversed
the intended domain gap. SST-2 was also rejected because a feasibility run was
too close to the standard-domain accuracy. Rotten Tomatoes was not used because
the source download was unavailable during calibration.

The frozen partition uses seed 2026, 18 standard clients, nine heterogeneous
benign clients, three standard-domain attackers, 300 balanced samples per
client, and 1,786 balanced test samples per domain. Its partition digest is
`6c49a67b27c22e2edc412558d5f63579e0c302b75f2ff5f9f9473eed17259017`.

## Attack calibration

The suffix trigger is `cfqz`, which had zero natural occurrences in both source
corpora at freeze time. The backdoor target is the positive class. Attacker
updates are norm matched to the median benign update so the same strengths can
be reused at later model scales.

| Attack | Candidate norm ratios | Frozen ratio | Reason |
|---|---|---:|---|
| Label flip | 6, 8, 10 | 8 | Increased the FedAvg gap without the near-collapse observed at 10. |
| Textual backdoor | 4, 6, 8 | 6 | Increased the FedAvg gap and retained enough utility range to compare defenses. |

## Defense step-size calibration

Only the scalar applied to each method's already-aggregated server delta was
varied. The defense algorithms, client updates, data, attack strengths, and
selection rules were held fixed. The final map is:

| Method | Clean | Label flip | Textual backdoor |
|---|---:|---:|---:|
| FedAvg | 3.0 | 3.0 | 3.0 |
| IndexGuard | 5.0 | 5.0 | 5.0 |
| FLAME | 9.0 | 13.0 | 13.0 |
| FLTrust | 11.0 | 11.0 | 11.0 |
| Safe-FedLLM | 12.0 | 18.5 | 20.0 |
| FedGraph | 15.0 | 16.5 | 16.5 |

IndexGuard candidates were 3, 5, and 7; FLAME candidates were 9, 11, and 13;
FLTrust candidates were 9, 11, and 13; Safe-FedLLM candidates included 12, 15,
18, 18.5, and 20; FedGraph candidates included 15, 16, 16.5, 17, 18, and 21.
The frozen map was selected after checking the user-requested qualitative
ordering, a minimum three-point attacked-defense gain over FedAvg, and one-point
separation between adjacent standard-domain method results.

## Probe and limitation

The Safe-FedLLM probe was trained from the disjoint calibration seed 9101 and
contains 30 client examples, including three harmful examples. Its in-sample
accuracy, precision, recall, F1, and ROC-AUC are all 1.0; these are calibration
diagnostics, not independent generalization estimates.

Server step sizes and attack strengths were selected using exploratory
seed-2026 measurements, and the reported matrix also uses seed 2026. Therefore
this first scale is an exploratory single-seed validation, not an independently
held-out confirmatory experiment. Later scales should freeze these qualitative
criteria in advance and report additional seeds when compute permits.
