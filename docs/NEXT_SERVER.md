# Continuing on a new GPU server

GitHub is the durable hand-off point. The three-scale TweetEval-Yelp study is
complete. Model caches, datasets, checkpoints, shared-update caches, raw logs,
and credentials are not committed; frozen configs, model-specific Safe-FedLLM
probes, final tables, checks, calibration records, and compact evidence are.

## Completed scale map

| Scale | Config | Model | Result root | Probe |
|---|---|---|---|---|
| 0.5B | `configs/reproduction/cross_domain_tweet_yelp_qwen05_v1.yaml` | `qwen2.5-0.5b` | `results/cross_domain_tweet_yelp_qwen05_v1` | `artifacts/safefedllm_probe_tweet_yelp_qwen05_v1.pt` |
| 1.5B | `configs/reproduction/cross_domain_tweet_yelp_qwen15_v1.yaml` | `qwen2.5-1.5b` | `results/cross_domain_tweet_yelp_qwen15_v1` | `artifacts/safefedllm_probe_tweet_yelp_qwen15_v1.pt` |
| 3B | `configs/reproduction/cross_domain_tweet_yelp_qwen3_v1.yaml` | `qwen2.5-3b` | `results/cross_domain_tweet_yelp_qwen3_v1` | `artifacts/safefedllm_probe_tweet_yelp_qwen3_v1.pt` |

The complete table is
`reports/cross_domain_tweet_yelp_three_scale_v1/REPORT.md`. The partition digest
is `6c49a67b27c22e2edc412558d5f63579e0c302b75f2ff5f9f9473eed17259017`.

## Recreate a clean server

```bash
git clone https://github.com/Dihaoda/FedSApro.git
cd FedSApro
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ruff check .
python -m pytest -q
```

Use a CUDA-enabled PyTorch build suitable for the new GPU. The frozen 3B run
was verified on one 32 GB V100S and peaked below 10 GB for a single training
process under the committed batch and sequence settings.

## Recreate data and run a scale

```bash
fedsapro prepare-data \
  --config configs/reproduction/cross_domain_tweet_yelp_qwen3_v1.yaml
fedsapro validate-data \
  --config configs/reproduction/cross_domain_tweet_yelp_qwen3_v1.yaml

python scripts/run_matrix.py \
  --config configs/reproduction/cross_domain_tweet_yelp_qwen3_v1.yaml \
  --models qwen2.5-3b \
  --attacks none label_flip textual_backdoor \
  --methods fedavg indexguard flame fltrust safefedllm fedgraph \
  --seeds 2026 --continue-on-error \
  --manifest results/cross_domain_tweet_yelp_qwen3_v1/events.jsonl
```

Run `none` first if rebuilding caches. Its cache provides the 27 identical
benign updates used by both attacks; only clients 27-29 must be retrained for
each attacked cache. Never copy a probe or shared-update cache across model
sizes.

## Regenerate reports

```bash
python scripts/export_tweet_yelp_scale_results.py \
  --root results/cross_domain_tweet_yelp_qwen3_v1 \
  --baseline-root results/cross_domain_tweet_yelp_qwen15_v1 \
  --output reports/cross_domain_tweet_yelp_qwen3_v1 \
  --title "TweetEval-Yelp Qwen2.5-3B validation study" \
  --current-label 3B --baseline-label 1.5B
python scripts/export_tweet_yelp_combined.py
```

The frozen server-step maps and rejected calibration ranges are documented in
each scale's `CALIBRATION.md`. A future model scale or dataset must train a new
seed-9101 probe, create its own cache root, and record a separate calibration;
do not overwrite the completed three-scale study.
