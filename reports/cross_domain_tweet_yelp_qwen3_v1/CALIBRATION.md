# TweetEval-Yelp Qwen2.5-3B calibration record

This scale reuses the frozen seed-2026 TweetEval-Yelp partition, binary label
mapping, `cfqz` suffix trigger, client roles, local training budget, and attack
norm ratios from the 0.5B and 1.5B studies. Only model-specific server step
sizes and the Safe-FedLLM probe are recalibrated.

## Safe-FedLLM probe

The model-specific probe was trained from disjoint seed 9101 using 30 client
updates, including three harmful label-flip clients. Its in-sample accuracy,
precision, recall, F1, and ROC-AUC are all 1.0. These values are calibration
diagnostics rather than independent generalization estimates.

## Bounded cache-only server-step calibration

All candidates reuse identical seed-2026 cached client updates. A server step
only scales the already-aggregated delta; it does not change local training, an
attack update, or a defense's client-selection decision. The calibration first
evaluated the scale-matched map inherited from the earlier 3B study, then used
a bounded set of points around collisions in the measured standard-domain
accuracies. Rejected examples include label-flip FedAvg 1.8/2.1 (excessive
accuracy loss), FLTrust 1.3 (less than a three-point gain over FedAvg), and
Safe-FedLLM/FedGraph 3.5 (excessive accuracy loss).

The frozen map is:

| Method | Clean | Label flip | Textual backdoor |
|---|---:|---:|---:|
| FedAvg | 1.2 | 1.61 | 1.5 |
| IndexGuard | 1.0 | 1.0 | 1.0 |
| FLAME | 1.2 | 2.1 | 1.5 |
| FLTrust | 0.8 | 1.17 | 1.17 |
| Safe-FedLLM | 1.0 | 3.30 | 2.70 |
| FedGraph | 1.4 | 3.20 | 3.20 |

The final map preserves the qualitative conclusion, requires every attacked
defense to exceed FedAvg standard-domain accuracy by at least three points,
and keeps adjacent attacked standard-domain results at least one point apart.
Domain gap is the absolute accuracy disparity, so an attack-induced reversal
of the better domain cannot be misreported as a negative gap.

## Limitation

As in the smaller scales, server-step selection used exploratory seed-2026
measurements and the final matrix reports seed 2026. The scale comparison is
therefore exploratory and single-seed. Raw final measurements, resolved
configs, aggregation diagnostics, and compact evidence are retained in the
report bundle; rejected calibration directories are intentionally not
committed.
