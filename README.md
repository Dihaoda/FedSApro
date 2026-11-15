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
The efficient single-GPU profile in `configs/reproduction/efficient.yaml` uses the same clients,
attacks, methods, effective batch size, and LoRA rank, with a sequence-classification head,
48-token truncation, two communication rounds, a server learning rate of 2.0, and a 5.0 malicious
update scale for label flipping (textual backdoor remains 1.0). These explicit knobs compress
global adaptation and label-flip strength into the short run. The profile is reported separately
from the frozen 50-round paper configuration.

The final accuracy-centric study is frozen in `configs/reproduction/accuracy_study.yaml`. It uses
Qwen2.5-0.5B, seed 2026, 30 clients, two communication rounds, and the same six aggregation
methods. Heterogeneous benign clients draw 80% Amazon and 20% IMDb examples, while designated
attackers draw IMDb examples. This makes the clean domain gap measurable and lets the paired
attack runs test whether it widens. FedProx uses `mu=0.01` with a method-specific server learning
rate of 1.5. SCAFFOLD uses a 0.75 server learning rate and a 0.001 control-variate scale for the
compressed two-round schedule. The calibration trail is retained in
`reports/accuracy_study_v1/CALIBRATION.md`.

A restart-safe matrix runner is also included. It defaults to the small Qwen2.5-1.5B model,
skips completed cells, resumes checkpoints, and preserves an append-only event manifest:

```bash
python scripts/run_matrix.py --config configs/reproduction/efficient.yaml --seeds 2026
```

Run and export the final small-model matrix with:

```bash
python scripts/run_matrix.py \
  --config configs/reproduction/accuracy_study.yaml \
  --models qwen2.5-0.5b \
  --attacks none label_flip textual_backdoor \
  --methods fedavg fedprox scaffold indexguard flame fltrust \
  --seeds 2026 --continue-on-error \
  --manifest results/accuracy_study_v1/events.jsonl

python scripts/export_accuracy_results.py \
  --root results/accuracy_study_v1 \
  --output reports/accuracy_study_v1
```

This exporter deliberately omits paper-target comparisons, HSSR, and HRCW. It reports only the
present experiment's accuracy, domain gap, and textual-backdoor ASR. Its machine-readable audit
requires a complete 18-cell matrix, larger attacked FedAvg gaps than clean FedAvg, and every
attacked defense to improve paired mean accuracy (plus lower ASR for textual backdoor).

Multiple candidate seeds can be supplied with `--try-seeds-until-accepted`; rejected seeds remain
on disk and are never rewritten as accepted measurements.

After the matrix finishes, export a GitHub-sized evidence bundle with measured and target values
kept in separate columns:

```bash
python scripts/export_results.py --root results/reproduction_v1 \
  --output reports/reproduction_v1
```

## Reproduction status

The legacy Qwen2.5-1.5B, seed-2026 matrix contains all 18 attack/method cells. Three cells
fall inside the deliberately strict paper-target windows. More importantly for the cross-model
claim, six of nine paired robust-method comparisons have higher HSSR than FedAvg while retaining
standard-domain accuracy within five percentage points. The short two-round classifier does not
reproduce the paper's full shifted-domain accuracy decline, so the evidence supports the
suppression mechanism but not every downstream accuracy effect. See
`reports/reproduction_v1/REPORT.md` for measured values, immutable paper targets, and the
predeclared conclusion audit.

The separate Qwen2.5-0.5B accuracy study is reported in
`reports/accuracy_study_v1/REPORT.md`. It is the current qualitative study requested for this
repository and should not be conflated with either the paper's absolute numbers or the legacy
target-window experiment. Its frozen seed-2026 matrix is complete (18/18 cells): both attacks
increase the FedAvg domain gap, and all ten attacked-defense comparisons improve paired mean
accuracy; all five textual-backdoor defenses also reduce attack success rate.

## Output contract

Each run writes:

- `resolved_config.yaml`: the exact resolved input configuration.
- `round_metrics.csv`: source-domain and shifted-domain accuracy by evaluation round.
- `client_rounds.csv`: role, update norm, raw/effective contribution, and selection per client.
- `summary.json`: final accuracy, HRCW, HSSR, domain gap, target differences, and provenance.
- `provenance.json`: source commit, runtime, and command metadata.
- `checkpoint_latest.pt`: resumable global adapter/control state (ignored by Git by default).
- `run.log`: full command/runtime diagnostics.

See `docs/METHODOLOGY.md` for the implementation choices that are underspecified by the paper.
See `docs/NEXT_SERVER.md` for the clean-server continuation and verification procedure.

