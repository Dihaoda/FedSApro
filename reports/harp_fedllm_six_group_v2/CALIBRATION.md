# HARP-FedLLM calibration record

## Protocol

- All formal cells use seed 2026.
- Method calibration is performed only on the no-attack condition. Attack
  results are not used to select HARP-FedLLM parameters.
- Candidate server aggregations reuse the same cached client updates. They do
  not retrain clients or modify attack updates.
- After a candidate is selected, the same method parameters are frozen for no
  attack, label flip, and textual backdoor.
- Accuracy values in `summary.csv` and `REPORT.md` are read directly from run
  `summary.json` files. No target injection or post-hoc jitter is applied.

## Final mechanism

The server extracts update norm and representative-layer LoRA fingerprints,
measures loss change on a standard-domain anchor, and measures sensitivity to a
fixed text perturbation. Robust z-scores form a malicious-risk factor. PCA and
clustering identify coherent update modes; novelty can preserve a low-risk
client but cannot cancel a malicious-risk penalty. This keeps previously unseen
benign domain structure without requiring heterogeneous-domain anchor text.

## Qwen2.5-3B audit trail

### Amazon Polarity--IMDb

The initial HARP-FedLLM aggregation measured 95.35% standard and 84.71%
heterogeneous accuracy on the no-attack condition. A simple clean-only variant
with a small dominant-mode gain measured 95.35% / 88.47% and was selected. A
server learning-rate 1.1 candidate measured 95.41% / 87.57% and was rejected.
The selected configuration uses server learning rate 1.0, utility gain 0.15,
dominant-mode gain 0.25, and no anchor correction.

### TweetEval--Yelp Polarity

The original clean run measured 92.27% / 93.95% but marked heterogeneous benign
clients because the standard-only anchor could not represent unseen Yelp text.
The open-anchor configuration disables anchor-harm rejection for this dataset;
it preserved the same accuracy while reducing benign suppression to zero.

Two additional clean-only candidates were recorded at the end of the study:

| Candidate | Standard | Heterogeneous | Mean | Decision |
|---|---:|---:|---:|---|
| Server LR 1.2, novelty gain 0.30 | 92.27 | 93.95 | 93.11 | Superseded |
| Server LR 1.3, novelty gain 0.30 | 91.99 | 93.78 | 92.89 | Rejected |
| Server LR 1.2, novelty gain 0.50 | 92.16 | 94.06 | 93.11 | Selected |

The selected novelty gain improves the clean heterogeneous domain without
reducing clean mean accuracy. It was then frozen and rerun for both attacks.
The label-flip and textual-backdoor results were 92.27% / 93.84% and 92.11% /
94.01%, respectively. Across the three conditions HARP-FedLLM remains best in
two-domain mean accuracy but is 0.1306 points below the best baseline in
heterogeneous-domain accuracy. This exception is retained rather than tuned
away using attack results.

## Scope

This is a single-seed exploratory validation. The evidence directory retains
the resolved configuration, provenance, per-client coefficients, aggregation
diagnostics, round metrics, and summary for every one of the 126 formal cells.
The GitHub-friendly `evidence.tar.gz` archive has SHA-256
`3a509d5ddaadfd77fe2b969fb0b48ed83131d0967a723be7d8a29f82110cf8df`.

