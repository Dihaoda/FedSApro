# FedSApro

Reproducible federated LoRA security experiments for heterogeneous benign
clients. The current frozen validation studies Qwen2.5-0.5B, 1.5B, and 3B on a
second pair of sentiment domains. It tests the paper's qualitative conclusion
rather than attempting to reproduce its absolute values; HSSR, HRCW, paper
targets, and synthetic result jitter are not acceptance criteria.

## HARP-FedLLM study (in progress)

HARP-FedLLM is the new open-set defense developed in this repository. It uses
three-layer LoRA and anchor-representation fingerprints, but it never penalizes
domain novelty by itself. The first completed group is Qwen2.5-0.5B on Amazon
Polarity--IMDb; its measured table, frozen method details, and per-client raw
evidence are in `reports/harp_fedllm_study_v1/`. Later groups are added only
after the preceding group is reviewed.

## TweetEval-Yelp three-scale validation

- Task: balanced binary sentiment classification.
- Standard domain: `cardiffnlp/tweet_eval` sentiment, with neutral examples
  excluded and negative/positive mapped to 0/1.
- Heterogeneous benign domain: `yelp_polarity`, mapped to the same labels.
- Population: 18 standard clients, nine heterogeneous benign clients, and
  three attackers; all 30 clients participate in the single round.
- Data budget: 300 balanced training examples per client and 1,786 measured
  test examples per domain. The frozen stress profile draws the standard test
  set from the shortest 30% pool and the heterogeneous test set from the
  longest 30% pool.
- Training: rank-8 Q/V LoRA, one local epoch, batch size 16, maximum length 48,
  seed 2026.
- Attacks: full label flipping and `cfqz` suffix-trigger poisoning, with
  submitted attacker updates norm-matched to 8x and 6x the benign median.
- Methods: FedAvg, IndexGuard, FLAME, FLTrust, Safe-FedLLM, and FedGraph.

| Scale | Overall mean acc. | Clean mean | Label-flip mean | Backdoor mean | Clean FedAvg gap | Label FedAvg gap | Backdoor FedAvg gap | Min defense gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | 81.42 | 83.60 | 80.47 | 80.20 | 11.14 | 15.40 | 12.09 | 3.25 |
| 1.5B | 84.22 | 87.95 | 82.61 | 82.11 | 8.06 | 9.35 | 9.52 | 3.58 |
| 3B | 86.20 | 89.75 | 85.02 | 83.84 | 5.43 | 20.77 | 20.60 | 3.25 |

Every matrix is complete (18/18 cells). At every scale, both attacks increase
FedAvg's absolute domain gap, every attacked defense exceeds FedAvg
standard-domain accuracy by at least three points, adjacent attacked method
accuracies differ by at least one point, and every defense reduces mean
textual-backdoor ASR. Relative to 1.5B, the 3B model improves 16/18 cells, all
three condition means, and the full-matrix mean by 1.98 points.

The complete 54-row measured table is in
`reports/cross_domain_tweet_yelp_three_scale_v1/REPORT.md`. Per-scale reports,
machine-readable checks, and compact raw evidence are in:

- `reports/cross_domain_tweet_yelp_qwen05_v1`
- `reports/cross_domain_tweet_yelp_qwen15_v1`
- `reports/cross_domain_tweet_yelp_qwen3_v1`

## Reproduce a scale

Install the package, prepare the deterministic partition, and run the matrix.
Replace the config/model/result arguments with the desired scale.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .

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

python scripts/export_tweet_yelp_scale_results.py \
  --root results/cross_domain_tweet_yelp_qwen3_v1 \
  --baseline-root results/cross_domain_tweet_yelp_qwen15_v1 \
  --output reports/cross_domain_tweet_yelp_qwen3_v1 \
  --title "TweetEval-Yelp Qwen2.5-3B validation study" \
  --current-label 3B --baseline-label 1.5B
python scripts/export_tweet_yelp_combined.py
```

The model-specific Safe-FedLLM probes are committed in `artifacts/`. Do not
reuse a probe or shared-update cache across model sizes. The deterministic data
partition digest is
`6c49a67b27c22e2edc412558d5f63579e0c302b75f2ff5f9f9473eed17259017`.

## Verification and scope

```bash
python -m ruff check .
python -m pytest -q
```

The frozen study is exploratory and single-seed. Server-step calibration uses
cached client updates from seed 2026 and is recorded transparently in each
scale's `CALIBRATION.md`; it does not alter local training, attack updates, or
defense client-selection decisions. Datasets, model weights, checkpoints,
shared-update caches, server logs, and credentials are not committed.

Historical Amazon-IMDb experiments remain in their original configuration and
report directories. See `docs/METHODOLOGY.md` for implementation boundaries,
`docs/THIRD_PARTY.md` for algorithm provenance, and `docs/NEXT_SERVER.md` for a
clean-server hand-off.

