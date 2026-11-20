# Methodology and reporting boundary

## Paired design

For a fixed model and seed, all methods use the same persisted client partition, base adapter
initialization, and local minibatch order. Local seeds depend only on `(seed, round, client_id)`.
Changing the aggregation defense therefore does not silently change the training examples.

The current security study uses one communication round and a server learning rate of 3.0 as an
explicit round-compression strategy. Every formal cell uses the same value. The study is intended
to test qualitative relationships under a small Qwen model, not reproduce the paper's absolute
numbers.

## Model and objective

The model is Qwen2.5-0.5B-Instruct with rank-8 LoRA on every attention query/value projection.
The transformer backbone and sequence-classification head are frozen; only LoRA parameters are
federated. The two classifier rows are initialized from the pretrained LM-head vectors of the
configured negative/positive label tokens. Freezing the head keeps the experiment a LoRA-only
federation and prevents a small randomly initialized head from dominating update similarity.

## Clients and evaluation stress profile

The 30-client population contains 24 standard Amazon clients, three heterogeneous clients with
50% Amazon and 50% IMDb data, and three attacker clients with IMDb data. Every client receives 300
balanced examples. The clean evaluation domains each contain 2,000 balanced examples.

The `length_extremes` evaluation profile chooses the shortest balanced Amazon examples and the
longest balanced IMDb examples. Selection uses only source domain, label, and character length; it
never queries the model or uses accuracy. This creates a fixed distribution-shift stress test.

## Attacks

The aggressive profile flips every label on malicious clients for `label_flip`. For
`textual_backdoor`, it appends the trigger `movie` to every malicious example and sets the target
to class 0. Submitted attacker updates are multiplied only on LoRA query/value slices: 5x for
label flipping and 3x for the textual backdoor. No-attack updates use scale 1.0.

These settings intentionally create a strong stress test. They preserve the attack mechanisms but
must not be described as the poisoning ratios or horizon used by the paper.

## Aggregation methods

- FedAvg uses sample-count weights.
- IndexGuard applies Top-K salience, index-overlap similarity, average-linkage clustering, a
  cohesion margin, and a minority-size check.
- FLAME applies HDBSCAN filtering and norm clipping; noise is disabled in this utility-focused
  comparison.
- FLTrust uses a 200-example clean server root set and three root epochs.
- Safe-FedLLM uses a held-out linear LoRA probe and the Shadow-Level security mapping.
- FedGraph builds a cosine-similarity graph, extracts degree/betweenness/closeness fingerprints,
  and applies the paper's HDBSCAN outlier rule.

FedProx and SCAFFOLD remain available for legacy profiles but are excluded from the current matrix
because they are optimization methods rather than attack-specific robust aggregators.

## Safe-FedLLM adaptation

The official Shadow-Level method can inspect a shadow LoRA trajectory. This one-round terminal-
delta harness exposes one submitted delta per client, so the implementation uses the malicious
probability of that terminal delta as the soft harmful ratio. The feature is the first transformer
layer's LoRA-B delta normalized to unit length with `log1p(norm)` appended. The security factor is
`max(0.01, (1-p)^7)`.

The probe is trained only on seed 9101 artifacts covering clean, label-flip, and textual-backdoor
calibration runs. The formal seed 2026 is explicitly excluded from probe fitting. The committed
artifact records its source hashes, seeds, feature policy, and in-sample metrics.

## FedGraph adaptation

The FedGraph paper uses full LoRA deltas and a default cosine threshold of 0.8. Those settings
produced an extremely sparse, non-separating graph on Qwen2.5-0.5B. On the independent seed-9101
calibration artifacts, the first-layer LoRA-B slice with threshold 0.50 separated all three
malicious clients with no benign false positives for both attacks. The formal configuration fixes
that input slice and threshold, then follows the paper's graph fingerprints and
`HDBSCAN(min_cluster_size=16, min_samples=1)` rule. `allow_single_cluster` is enabled so a clean
round can retain a single benign cluster.

## Reporting boundary

The final exporter reads measured outputs only. It reports standard accuracy, heterogeneous
accuracy, their mean and gap, textual-backdoor ASR, and paired differences from FedAvg. Paper
target values, HSSR, and HRCW are deliberately absent. A result is a single-seed stress-test result,
not a multi-seed estimate.

Rejected calibration attempts and superseded implementations remain documented rather than being
silently overwritten. The GitHub evidence bundle contains each formal cell's summary, resolved
configuration, provenance, client coefficients, round metrics, and aggregation diagnostics; model
weights and dataset caches are excluded.
