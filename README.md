# HARP-FedLLM

Official code repository for **HARP-FedLLM**.

## Repository structure

- `artifacts/`: model-specific probe files used by the baselines.
- `configs/`: base settings and the six experiment configurations.
- `docs/`: methodology, baseline provenance, and reproducibility notes.
- `reports/`: complete experimental results and supporting evidence.
- `scripts/`: experiment launchers, probe construction, and result export tools.
- `src/fedsapro/`: federated training, attacks, baselines, and HARP-FedLLM.
- `tests/`: unit tests for configurations, data, aggregation, and experiments.

## Quick start

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_six_group_study.py --dry-run
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full experiment
commands and [`reports/harp_fedllm_six_group_v2/REPORT.md`](reports/harp_fedllm_six_group_v2/REPORT.md)
for the measured results.
