# FedSApro

Reproduction code for *Heterogeneous Benign Client Suppression in Robust Federated LoRA*.
The repository implements one paired experiment harness for all paper methods so that model
initialization, client partitions, minibatch order, attacks, and evaluation sets can be held fixed.

## Paper experiment matrix

- Models: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct.
- Conditions: no attack, 30% label flipping, 10% textual backdoor.
- Methods: FedAvg, FedProx, SCAFFOLD, IndexGuard, FLAME, and FLTrust.
- Population: 30 clients for 50 full-participation rounds. Eighteen standard benign clients use
  Amazon Polarity, six heterogeneous benign clients use 60% Amazon and 40% IMDb, and six
  designated attacker clients use Amazon data (remaining benign in the no-attack condition).
- Training: 500 balanced samples per client, one local epoch, AdamW at `1e-4`, batch size 16,
  sequence length 256, and rank-8 LoRA on every attention query/value projection.
- Evaluation: disjoint balanced Amazon and IMDb test sets with 2,000 examples each.

The main diagnostics are heterogeneous relative contribution weight (HRCW) and heterogeneous
strong suppression rate (HSSR, threshold `tau=0.25`). Every accepted result remains linked to its
configuration, seed, raw round/client logs, checkpoint, and source commit. A single-seed result is
reported as a single-seed reproduction and is not represented as the paper's three-seed mean.

## Quick start

```bash
python -m pip install -r requirements.txt
fedsapro prepare-data --config configs/paper/base.yaml
fedsapro run --config configs/paper/base.yaml --method fedavg --attack none \
  --model qwen2.5-1.5b --seed 2026
fedsapro summarize --run-dir results/qwen2.5-1.5b/none/fedavg/seed_2026
```

For a fast installation/model smoke test, override rounds, client count, samples, and evaluation
size without modifying the frozen paper config:

```bash
fedsapro run --config configs/paper/base.yaml --method fedavg --attack none \
  --model qwen2.5-1.5b --seed 2026 --smoke
```

Pilot/calibration changes are applied as explicit dotted overrides and are captured in the
resolved configuration, for example `--set federated.rounds=5 --set federated.eval_every=1`.

## Output contract

Each run writes:

- `resolved_config.yaml`: the exact resolved input configuration.
- `round_metrics.csv`: source-domain and shifted-domain accuracy by evaluation round.
- `client_rounds.csv`: role, update norm, raw/effective contribution, and selection per client.
- `summary.json`: final accuracy, HRCW, HSSR, domain gap, target differences, and provenance.
- `checkpoint_latest.pt`: resumable global adapter/control state (ignored by Git by default).
- `run.log`: full command/runtime diagnostics.

See `docs/METHODOLOGY.md` for the implementation choices that are underspecified by the paper.

