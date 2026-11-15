# Continuing on a new GPU server

The repository is the durable hand-off point. Datasets, Hugging Face model files, run
checkpoints, and Python environments are intentionally not committed.

## 1. Clone and install

```bash
git clone https://github.com/Dihaoda/FedSApro.git
cd FedSApro
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use a CUDA-capable PyTorch build suitable for the new server if the resolver does not install
one automatically. The final accuracy study uses `Qwen/Qwen2.5-0.5B-Instruct` and normally fits
on a single modest GPU.

## 2. Recreate caches

```bash
fedsapro prepare-data --config configs/reproduction/accuracy_study.yaml
```

The first run downloads the model through Hugging Face. Once both the dataset and model are
cached, offline execution can be enabled with:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 3. Validate the checkout

```bash
python -m ruff check src tests scripts
python -m pytest -q
```

## 4. Reproduce the frozen 18-cell matrix

The final study is deliberately single-seed (`2026`) and small-model only. The runner skips
cells that already contain a summary and appends lifecycle events to its manifest.

```bash
export FEDSA_GITHUB_COMMIT="$(git rev-parse HEAD)"

python scripts/run_matrix.py \
  --config configs/reproduction/accuracy_study.yaml \
  --models qwen2.5-0.5b \
  --attacks none label_flip textual_backdoor \
  --methods fedavg fedprox scaffold indexguard flame fltrust \
  --seeds 2026 \
  --continue-on-error \
  --manifest results/accuracy_study_v1/events.jsonl
```

If the process is interrupted, run the same command again. Completed cells are preserved; cells
with a checkpoint resume from `checkpoint_latest.pt`.

## 5. Export and audit

```bash
python scripts/export_accuracy_results.py \
  --root results/accuracy_study_v1 \
  --output reports/accuracy_study_v1

python - <<'PY'
import json
from pathlib import Path

checks = json.loads(
    Path("reports/accuracy_study_v1/conclusion_checks.json").read_text()
)
assert checks["matrix_complete"]
assert checks["all_attacks_increase_fedavg_gap"]
assert checks["all_attacked_defenses_better_than_fedavg"]
print("accuracy-study conclusion checks passed")
PY
```

The table intentionally excludes HSSR, HRCW, and comparisons with the paper's absolute
numbers. Raw checkpoints and caches remain local; the compact report and evidence bundle are
the artifacts intended for GitHub.

