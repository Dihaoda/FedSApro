# Continuing on a new GPU server

GitHub is the durable hand-off point. Model caches, datasets, and resumable checkpoints are not
committed; the Safe-FedLLM probe, frozen configuration, final tables, and compact evidence are.

## 1. Clone and install

```bash
git clone https://github.com/Dihaoda/FedSApro.git
cd FedSApro
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use a CUDA-enabled PyTorch build appropriate for the server. Qwen2.5-0.5B fits on a single 32 GB
GPU with substantial margin under the frozen batch/sequence settings.

## 2. Recreate deterministic data

```bash
fedsapro prepare-data \
  --config configs/reproduction/aggressive_security_study.yaml

fedsapro validate-data \
  --config configs/reproduction/aggressive_security_study.yaml
```

The first command downloads Amazon Polarity and IMDb when they are not cached. The client split and
length-based test profile are deterministic from seed 2026.

## 3. Validate the checkout

```bash
test -f artifacts/safefedllm_probe_qwen05.pt
python -m ruff check .
python -m pytest -q
```

## 4. Run or resume the 18-cell matrix

```bash
export FEDSA_GITHUB_COMMIT="$(git rev-parse HEAD)"

python scripts/run_matrix.py \
  --config configs/reproduction/aggressive_security_study.yaml \
  --models qwen2.5-0.5b \
  --attacks none label_flip textual_backdoor \
  --methods fedavg indexguard flame fltrust safefedllm fedgraph \
  --seeds 2026 --continue-on-error \
  --manifest results/aggressive_security_study_v1/events.jsonl
```

Rerun the same command after an interruption. Complete summaries are skipped and incomplete cells
resume from `checkpoint_latest.pt`.

## 5. Export and verify

```bash
python scripts/export_aggressive_results.py \
  --root results/aggressive_security_study_v1 \
  --output reports/aggressive_security_study_v1

python - <<'PY'
import json
from pathlib import Path

path = Path("reports/aggressive_security_study_v1/conclusion_checks.json")
checks = json.loads(path.read_text())
assert checks["matrix_complete"]
assert checks["all_predeclared_checks_pass"]
print("all formal checks passed")
PY
```

Commit the regenerated `reports/aggressive_security_study_v1` directory. Do not commit model
caches, datasets, checkpoints, or Hugging Face credentials.
