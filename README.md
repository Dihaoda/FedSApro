# FedSApro

Reproducible federated LoRA security experiments for heterogeneous benign clients. The current
study is an accuracy-, domain-gap-, and attack-success-rate experiment on Qwen2.5-0.5B. It is not
presented as a reproduction of the paper's absolute values, and its final report deliberately
excludes paper targets, HSSR, and HRCW.

## Current frozen study

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, rank-8 LoRA on attention query/value projections.
- Federated parameters: LoRA only. The two-class head is initialized from label-token LM-head
  rows and then frozen.
- Population: 30 full-participation clients: 24 Amazon clients, three 50/50 Amazon/IMDb
  heterogeneous clients, and three IMDb attacker clients.
- Data: 300 balanced examples per client. Evaluation uses 2,000 balanced examples per domain;
  the stress profile selects short Amazon and long IMDb reviews using only labels and text length.
- Training: one communication round, one local epoch, batch size 16, maximum length 48, client
  learning rate `1e-4`, and server learning rate `3.0`.
- Attacks: full label flipping with a 5x LoRA-only submitted-update scale, and full suffix-trigger
  poisoning with a 3x scale. These are explicit aggressive stress-test settings.
- Methods: FedAvg, IndexGuard, FLAME, FLTrust, Safe-FedLLM, and FedGraph. FedProx and SCAFFOLD are
  intentionally excluded from this matrix.
- Formal seed: `2026`. Safe-FedLLM and FedGraph calibration uses the disjoint seed `9101`.

Safe-FedLLM is a documented terminal-delta adaptation of its Shadow-Level defense. Its held-out
linear probe consumes the normalized first-layer LoRA-B delta plus its log norm and maps the
malicious probability to the paper's `(1-p)^7` security factor. FedGraph follows the paper's
cosine graph, degree/betweenness/closeness fingerprint, and HDBSCAN outlier filtering. For this
small LoRA model, its graph uses the first-layer LoRA-B slice and a threshold of 0.50 selected on
the disjoint calibration seed.

## Result status

The frozen single-seed matrix is complete (18/18 cells), and all predeclared conclusion checks
pass. Clean FedAvg has a 17.50-point domain gap. Under FedAvg, label flipping increases the gap to
31.45 points and textual backdoor poisoning increases it to 35.30 points. All ten attacked-defense
cells improve mean accuracy by at least 8.72 points, heterogeneous accuracy by at least 13.60
points, and reduce the attacked gap by at least 9.75 points. Every textual-backdoor defense reduces
mean ASR by at least 38.80 points.

See `reports/aggressive_security_study_v1/REPORT.md` for the complete table,
`conclusion_checks.json` for exact pass/fail decisions, and `evidence/` for each cell's resolved
configuration, raw metrics, coefficients, provenance, and aggregation diagnostics. Calibration
choices and rejected attempts are recorded in `CALIBRATION.md`.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pip install -e .

fedsapro prepare-data \
  --config configs/reproduction/aggressive_security_study.yaml

python scripts/run_matrix.py \
  --config configs/reproduction/aggressive_security_study.yaml \
  --models qwen2.5-0.5b \
  --attacks none label_flip textual_backdoor \
  --methods fedavg indexguard flame fltrust safefedllm fedgraph \
  --seeds 2026 --continue-on-error \
  --manifest results/aggressive_security_study_v1/events.jsonl

python scripts/export_aggressive_results.py \
  --root results/aggressive_security_study_v1 \
  --output reports/aggressive_security_study_v1
```

The matrix runner skips complete cells and resumes incomplete checkpoints. The committed
`artifacts/safefedllm_probe_qwen05.pt` is the calibration-seed probe required by Safe-FedLLM.

## Verification

```bash
python -m ruff check .
python -m pytest -q

python - <<'PY'
import json
from pathlib import Path

checks = json.loads(
    Path("reports/aggressive_security_study_v1/conclusion_checks.json").read_text()
)
assert checks["matrix_complete"]
assert checks["all_predeclared_checks_pass"]
print("aggressive security-study checks passed")
PY
```

The repository retains the earlier paper-target and accuracy-study profiles for historical
comparison. They are separate experiments and must not be mixed with the current report.

See `docs/METHODOLOGY.md` for implementation boundaries, `docs/THIRD_PARTY.md` for algorithm
provenance, and `docs/NEXT_SERVER.md` for a clean-server hand-off.
