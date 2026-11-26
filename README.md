# HARP-FedLLM

Reference implementation and measured artifacts for **HARP-FedLLM**, a
heterogeneity-aware robust aggregation method for federated LoRA fine-tuning.
The method protects coherent but previously unseen benign client domains while
down-weighting label-flip and textual-backdoor updates.

## Overview

HARP-FedLLM combines four server-side signals:

1. LoRA update norms and representative-layer fingerprints;
2. loss change on a small standard-domain server anchor;
3. sensitivity to a fixed text perturbation; and
4. coherent update modes discovered among low-risk clients.

Domain novelty is used only to preserve a low-risk client. It is never a
standalone rejection signal, so the server does not require examples from an
unknown heterogeneous domain.

## Formal study

The frozen evaluation contains 126 measured cells:

- three Qwen2.5 scales: 0.5B, 1.5B, and 3B;
- two domain pairs: Amazon Polarity--IMDb and TweetEval--Yelp Polarity;
- three conditions: no attack, label flip, and textual backdoor; and
- seven aggregators: FedAvg, IndexGuard, FLAME, FLTrust, Safe-FedLLM,
  FedGraph, and HARP-FedLLM.

Every cell uses seed 2026, 30 participating clients (18 standard, nine
heterogeneous benign, and three attackers), 300 balanced examples per client,
rank-8 Q/V LoRA, one local epoch, 48-token local training/fingerprints, and
128-token evaluation. Within each model/dataset/attack group, all methods use
the same cached client updates.

### HARP-FedLLM averages across the three conditions

| Model | Domains | Standard | Heterogeneous | Two-domain mean | Heterogeneous margin | Mean margin |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | Amazon--IMDb | 93.88 | 83.09 | 88.48 | +0.19 | +0.10 |
| Qwen2.5-0.5B | TweetEval--Yelp | 89.59 | 89.25 | 89.42 | +0.54 | +0.22 |
| Qwen2.5-1.5B | Amazon--IMDb | 95.95 | 87.25 | 91.60 | +0.19 | +0.01 |
| Qwen2.5-1.5B | TweetEval--Yelp | 92.48 | 93.00 | 92.74 | +0.15 | +0.11 |
| Qwen2.5-3B | Amazon--IMDb | 95.43 | 88.56 | 91.99 | +0.09 | +0.08 |
| Qwen2.5-3B | TweetEval--Yelp | 92.18 | 93.97 | 93.08 | -0.13 | +0.03 |

HARP-FedLLM has the highest two-domain mean accuracy in all six groups and the
highest heterogeneous-domain average in five of six groups. The disclosed
exception is Qwen2.5-3B on TweetEval--Yelp: its mean accuracy remains first,
while its heterogeneous-domain average trails FLTrust by 0.13 points.

The [complete table](reports/harp_fedllm_six_group_v2/REPORT.md),
[126-row CSV](reports/harp_fedllm_six_group_v2/summary.csv), machine-readable
checks, calibration record, and compressed per-cell evidence are committed.
Reported values are read directly from run summaries; no paper-target injection
or post-hoc jitter is used.

## Repository layout

```text
artifacts/                         Safe-FedLLM probes and metadata
configs/paper/                     Base schema and reference targets
configs/reproduction/              Six frozen formal configurations
docs/METHODOLOGY.md                Method and evaluation boundary
docs/REPRODUCIBILITY.md            Clean-machine reproduction guide
docs/THIRD_PARTY.md                Baseline provenance and adaptations
reports/harp_fedllm_six_group_v2/  Final tables, checks, and evidence archive
scripts/run_matrix.py              Restart-safe single-group matrix runner
scripts/run_six_group_study.py     Canonical six-group launcher
scripts/export_harp_six_group_results.py
src/fedsapro/                      Experiment and aggregation implementation
tests/                             CPU unit tests
```

## Installation

Python 3.10 or later and a CUDA-enabled PyTorch installation are recommended.

```bash
git clone https://github.com/Dihaoda/FedSApro.git
cd FedSApro
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

Model weights and datasets are downloaded from their original providers and
are not stored in this repository.

## Reproduction

Run one group:

```bash
fedsapro prepare-data \
  --config configs/reproduction/cross_domain_tweet_yelp_qwen3_v1.yaml

python scripts/run_matrix.py \
  --config configs/reproduction/cross_domain_tweet_yelp_qwen3_v1.yaml \
  --models qwen2.5-3b \
  --attacks none label_flip textual_backdoor \
  --methods fedavg indexguard flame fltrust safefedllm fedgraph harp_fedllm \
  --seeds 2026 --continue-on-error
```

Run all six groups with the canonical launcher:

```bash
python scripts/run_six_group_study.py --prepare-data
python scripts/export_harp_six_group_results.py
```

The launcher is restart-safe: completed cells are detected from their
`summary.json` files. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the
configuration map, artifact policy, expected output roots, and evidence checks.

## Scope

This repository reports a single-seed, one-round controlled study intended to
test the paper's qualitative conclusions under a bounded compute budget. It is
not a multi-seed uncertainty estimate and does not claim to reproduce absolute
numbers from a different model family or training horizon. HARP-FedLLM
calibration uses only the no-attack condition; selected parameters are frozen
for both attacks.

## Citation

The formal paper citation will be added after publication metadata is
available. Until then, cite this repository and the exact Git commit used for
an experiment.


