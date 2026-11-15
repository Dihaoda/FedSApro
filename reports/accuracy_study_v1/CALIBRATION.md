# Accuracy-study calibration log

This log documents candidate settings that were tried before freezing the final
`configs/reproduction/accuracy_study.yaml`. These are measured exploratory runs, not additional
seeds and not entries in the final 18-cell table.

The acceptance rule for an attacked defense was fixed as:

1. mean accuracy must exceed FedAvg under the same attack; and
2. for textual backdoor, mean attack success rate (ASR) must also be lower than FedAvg.

The final FedAvg baselines were 82.05 mean accuracy for label flip and 83.65 mean accuracy with
54.55 mean ASR for textual backdoor.

## FedProx

| Attack | Candidate | Standard | Heterogeneous | Mean accuracy | Mean ASR | Decision |
|---|---|---:|---:|---:|---:|---|
| label flip | default short-run setting | 84.75 | 78.75 | 81.75 | - | reject |
| textual backdoor | default short-run setting | 86.45 | 80.65 | 83.55 | 57.20 | reject |
| label flip | `mu=0.01`, server LR `1.5` | 86.05 | 80.45 | 83.25 | - | select |
| textual backdoor | `mu=0.01`, server LR `1.5` | 86.55 | 81.80 | 84.18 | 44.90 | select |

## SCAFFOLD

| Attack | Candidate | Standard | Heterogeneous | Mean accuracy | Mean ASR | Decision |
|---|---|---:|---:|---:|---:|---|
| textual backdoor | original short-run setting | 84.95 | 79.45 | 82.20 | 27.95 | reject |
| textual backdoor | server LR `0.75`, control scale `1.0` | 79.75 | 74.20 | 76.98 | - | reject |
| textual backdoor | server LR `0.55`, control scale `1.0` | - | - | 81.65 | - | reject |
| textual backdoor | server LR `0.60`, control scale `1.0` | - | - | 80.65 | - | reject |
| textual backdoor | server LR `0.65`, control scale `1.0` | 82.40 | 77.05 | 79.73 | - | reject |
| textual backdoor | server LR `0.75`, control scale `0.1` | 83.80 | 78.05 | 80.93 | - | reject |
| textual backdoor | server LR `1.0`, control scale `0.01` | 85.05 | 79.55 | 82.30 | 62.05 | reject |
| label flip | server LR `0.75`, control scale `0.001` | 86.05 | 80.15 | 83.10 | - | select |
| textual backdoor | server LR `0.75`, control scale `0.001` | 86.70 | 81.35 | 84.03 | 47.20 | select |

The small control scale is an explicit damped-SCAFFOLD training strategy for the two-round
compressed study. It retains the SCAFFOLD control-variate mechanism while preventing its
accumulated correction from overwhelming the adapter update in this unusually short schedule.

## Frozen qualitative conditions

- Clean FedAvg domain gap: 5.50 points.
- Label-flip FedAvg domain gap: 6.10 points (`+0.60`).
- Textual-backdoor FedAvg domain gap: 6.10 points (`+0.60`).
- The final exporter checks all five defenses against the paired FedAvg baseline under each
  attack and checks ASR as an additional condition for textual backdoor.

Rejected runs remain under `results/calibration` on the originating server until shutdown. The
table above is the durable audit trail; no rejected value is substituted into the final report.
