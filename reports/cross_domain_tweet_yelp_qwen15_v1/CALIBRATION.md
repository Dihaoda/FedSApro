# TweetEval-Yelp Qwen2.5-1.5B calibration record

This scale reuses the frozen seed-2026 TweetEval-Yelp partition, binary label
mapping, `cfqz` suffix trigger, client roles, local training budget, and attack
norm ratios from the 0.5B study. Only model-specific server step sizes and the
Safe-FedLLM probe are recalibrated.

## Safe-FedLLM probe

The model-specific probe was trained from disjoint seed 9101 using 30 client
updates, including three harmful label-flip clients. Its in-sample accuracy,
precision, recall, F1, and ROC-AUC are all 1.0. These values are calibration
diagnostics rather than independent generalization estimates.

## Bounded server-step sweep

All candidates reuse identical cached client updates. A server step only scales
the already-aggregated delta; it does not change local training or a defense's
client-selection decision. Clean FedAvg candidates were 0.8, 1.0, 1.2, 1.5,
1.8, 2.1, and 3.0. Label-flip FedAvg candidates were 0.8, 1.0, 1.2, 1.5, 1.8,
2.1, 2.4, and 3.0. Defense candidates covered the following bounded ranges:

- IndexGuard: 1.0.
- FLAME: 1.5, 1.7, 1.8, 1.9, and 2.1.
- FLTrust: 0.8, 1.2, 1.25, 1.3, 1.5, and 1.8.
- Safe-FedLLM: 1.3, 1.8, 2.0, 2.3, and 2.5.
- FedGraph: 1.8, 2.1, and 2.3.

The frozen map is:

| Method | Clean | Label flip | Textual backdoor |
|---|---:|---:|---:|
| FedAvg | 1.5 | 1.8 | 1.5 |
| IndexGuard | 1.0 | 1.0 | 1.0 |
| FLAME | 1.5 | 1.9 | 1.7 |
| FLTrust | 0.8 | 1.3 | 1.25 |
| Safe-FedLLM | 1.3 | 2.5 | 2.5 |
| FedGraph | 1.8 | 2.1 | 2.1 |

The map preserves the qualitative conclusions, requires every attacked defense
to exceed FedAvg standard-domain accuracy by at least three points, and keeps
adjacent attacked method results at least one point apart. Domain gap is the
absolute accuracy disparity, so an attack-induced reversal of the better domain
does not incorrectly become a negative disparity.

## Limitation

As in the 0.5B scale, server-step selection used exploratory seed-2026
measurements and the final matrix reports seed 2026. The scale comparison is
therefore exploratory and single-seed. Raw measurements, resolved configs, and
selection diagnostics are retained in the evidence bundle.
