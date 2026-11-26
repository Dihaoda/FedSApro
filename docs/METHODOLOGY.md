# Methodology

## Experimental question

Robust federated aggregation can mistake a coherent but unfamiliar benign
domain for an attack. The study asks whether attack suppression can be retained
without discarding those heterogeneous benign clients.

The evaluation compares methods on both a standard domain and a heterogeneous
benign domain. The primary utility statistic is the arithmetic mean of the two
accuracies; heterogeneous-domain accuracy is reported separately so that a
method cannot hide benign-domain suppression behind high standard-domain
accuracy.

## Paired design

For a fixed model, dataset pair, attack, and seed, every aggregation method uses
the same deterministic client partition, base adapter initialization, local
minibatch order, and submitted client updates. Local seeds depend only on
`(seed, round, client_id)`. Aggregator comparisons therefore change the server
step without silently changing local training data.

The formal study uses one communication round as an explicit compute-bounded
round-compression setting. Server learning rates and attack norm targets are
declared in the six committed configurations. Local training and defense
fingerprints use 48 tokens; final evaluation uses 128 tokens.

## Models and clients

The study uses Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, and
Qwen2.5-3B-Instruct with rank-8 LoRA on attention query/value projections. The
transformer backbone and classification head are frozen; only LoRA parameters
are federated. Classifier rows are initialized from the pretrained LM-head
vectors of the configured negative and positive label tokens.

Each group contains 30 fully participating clients: 18 standard-domain clients,
nine heterogeneous benign clients, and three attacker clients. Every client
receives 300 balanced examples. Each evaluation domain contains 1,786 measured
examples selected by the frozen length-pool stress profile.

## Domain pairs

- **Amazon Polarity--IMDb:** Amazon Polarity is the standard domain and IMDb is
  the heterogeneous benign domain.
- **TweetEval--Yelp Polarity:** binary TweetEval sentiment is the standard
  domain; neutral examples are excluded. Yelp Polarity is the heterogeneous
  benign domain and is mapped to the same two labels.

The server anchor contains standard-domain text only. HARP-FedLLM never assumes
that examples from the previously unseen heterogeneous domain are available at
the server.

## Attacks

- **Label flip:** all labels on malicious clients are flipped.
- **Textual backdoor:** a fixed suffix trigger is inserted into malicious
  examples and their labels are changed to the configured target class.

Submitted attacker updates are norm-matched according to the model- and
dataset-specific frozen configuration. This retains the attack mechanisms while
making their strengths explicit and reproducible.

## Aggregation methods

- **FedAvg** uses sample-count weights.
- **IndexGuard** applies Top-K salience, index-overlap similarity,
  average-linkage clustering, a cohesion margin, and a minority-size check.
- **FLAME** uses HDBSCAN filtering and norm clipping. Noise is disabled in this
  utility-focused comparison.
- **FLTrust** weights updates using a clean standard-domain server root.
- **Safe-FedLLM** applies a model-specific LoRA probe and soft security weights.
- **FedGraph** builds a cosine graph and applies graph-centrality fingerprints
  with HDBSCAN outlier detection.
- **HARP-FedLLM** combines open-set risk scoring with coherent-mode
  preservation.

FedProx and SCAFFOLD remain implemented in the harness but are excluded from the
formal matrix because they are federated optimization methods rather than
attack-specific robust aggregators.

## HARP-FedLLM

For each client update, HARP-FedLLM records its log norm and representative LoRA
layer outputs. It measures standard-anchor loss change and sensitivity to a
fixed text perturbation. Robust medians and scales turn these measurements into
norm, harm, and sensitivity excess scores; their sum defines malicious risk.
The risk is mapped to a bounded soft security factor.

Separately, PCA and clustering expose coherent update modes among clients that
already pass the risk gate. A sufficiently large, separated mode can be
preserved even when it differs from the dominant standard-domain mode. Domain
novelty may increase the contribution of a low-risk client, but it cannot
override a malicious-risk penalty. This ordering is the core protection against
rewarding an unusual attack merely because it is novel.

Dataset-specific gains are selected on the no-attack condition and then frozen
for label flip and textual backdoor. Accepted and rejected choices are recorded
in `reports/harp_fedllm_six_group_v2/CALIBRATION.md`.

## Safe-FedLLM and FedGraph adaptations

Safe-FedLLM's model-specific probes are trained on independent seed-9101
calibration artifacts. Formal seed 2026 is excluded. Each committed JSON file
records feature policy, source hashes, and calibration metadata.

The FedGraph implementation follows the cosine graph, graph-centrality, and
HDBSCAN structure. Its LoRA feature scope and thresholds are explicit in each
formal configuration. These adaptations are disclosed because the harness sees
one terminal LoRA delta per client rather than a long shadow trajectory.

## Reporting boundary

The exporter reads measured `summary.json` files and reports standard accuracy,
heterogeneous accuracy, their mean and gap, and textual-backdoor ASR. It does
not replace outputs with paper targets. HSSR and HRCW remain diagnostic fields
inside run evidence but are not acceptance criteria.

The study is a single-seed stress test, not a confidence interval. The evidence
archive retains each formal cell's resolved configuration, provenance,
per-client coefficients, aggregation diagnostics, round metrics, and summary.
Model weights, datasets, checkpoints, and shared-update caches are excluded.


