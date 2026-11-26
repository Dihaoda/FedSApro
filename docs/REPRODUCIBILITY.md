# Reproducibility guide

## Environment

- Linux with Python 3.10 or later;
- a CUDA-enabled PyTorch build for full experiments;
- approximately 32 GB of GPU memory for the frozen 3B configuration; and
- network access for the first model and dataset download.

Install the package and development checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

## Frozen study map

| Study key | Model | Configuration | Output root | Safe-FedLLM probe |
|---|---|---|---|---|
| `amazon_imdb_qwen05` | Qwen2.5-0.5B | `configs/reproduction/natural_domain_attack_study.yaml` | `results/natural_domain_attack_study_v4_eval128` | `artifacts/safefedllm_probe_qwen05.pt` |
| `amazon_imdb_qwen15` | Qwen2.5-1.5B | `configs/reproduction/natural_domain_attack_study_qwen15.yaml` | `results/natural_domain_attack_study_qwen15_v6_eval128` | `artifacts/safefedllm_probe_qwen15_v4.pt` |
| `amazon_imdb_qwen3` | Qwen2.5-3B | `configs/reproduction/natural_domain_attack_study_qwen3_v1.yaml` | `results/natural_domain_attack_study_qwen3_v2_eval128` | `artifacts/safefedllm_probe_qwen3_v1.pt` |
| `tweet_yelp_qwen05` | Qwen2.5-0.5B | `configs/reproduction/cross_domain_tweet_yelp_qwen05_v1.yaml` | `results/cross_domain_tweet_yelp_qwen05_v2_eval128` | `artifacts/safefedllm_probe_tweet_yelp_qwen05_v1.pt` |
| `tweet_yelp_qwen15` | Qwen2.5-1.5B | `configs/reproduction/cross_domain_tweet_yelp_qwen15_v1.yaml` | `results/cross_domain_tweet_yelp_qwen15_v2_eval128` | `artifacts/safefedllm_probe_tweet_yelp_qwen15_v1.pt` |
| `tweet_yelp_qwen3` | Qwen2.5-3B | `configs/reproduction/cross_domain_tweet_yelp_qwen3_v1.yaml` | `results/cross_domain_tweet_yelp_qwen3_v2_eval128` | `artifacts/safefedllm_probe_tweet_yelp_qwen3_v1.pt` |

Never reuse a Safe-FedLLM probe or shared-update cache across model sizes.

## Run the study

Preview the commands without starting training:

```bash
python scripts/run_six_group_study.py --dry-run
```

Prepare deterministic partitions and run all 126 cells:

```bash
python scripts/run_six_group_study.py --prepare-data
```

Run one or more groups:

```bash
python scripts/run_six_group_study.py \
  --studies amazon_imdb_qwen05 tweet_yelp_qwen05 \
  --prepare-data
```

The launcher delegates to `scripts/run_matrix.py`. A completed cell is skipped
when its `summary.json` has the configured final round; interrupted cells resume
from `checkpoint_latest.pt` when available. Events are appended to a manifest
under `results/manifests/`.

## Export the formal report

After all six output roots are present:

```bash
python scripts/export_harp_six_group_results.py
```

The exporter requires 126 summaries and produces:

- `REPORT.md`: human-readable full tables;
- `summary.csv` and `summary.json`: machine-readable results;
- `conclusion_checks.json`: per-condition and group-level comparisons; and
- `evidence/`: compact provenance copied from each cell.

The committed `evidence.tar.gz` contains that evidence directory. Verify it
before unpacking:

```bash
sha256sum reports/harp_fedllm_six_group_v2/evidence.tar.gz
# 3a509d5ddaadfd77fe2b969fb0b48ed83131d0967a723be7d8a29f82110cf8df

tar -xzf reports/harp_fedllm_six_group_v2/evidence.tar.gz \
  -C reports/harp_fedllm_six_group_v2
```

## Artifact policy

The repository commits source, frozen configurations, small Safe-FedLLM probes,
formal tables, and compact evidence. It intentionally excludes datasets, model
weights, checkpoints, full result trees, logs, and shared-update caches. No
credentials or server-specific paths are required.


