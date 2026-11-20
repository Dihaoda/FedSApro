from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the held-out linear LoRA-delta probe for Safe-FedLLM Shadow-Level"
    )
    parser.add_argument("inputs", nargs="+", help="Calibration run directories or feature files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--c", type=float, default=100.0)
    return parser.parse_args()


def _files(inputs: list[str]) -> list[Path]:
    out: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            out.update(path.rglob("safefedllm_features_round_*.pt"))
        elif path.is_file():
            out.add(path)
        else:
            raise FileNotFoundError(path)
    if not out:
        raise RuntimeError("No Safe-FedLLM calibration feature artifacts found")
    return sorted(out)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = _arguments()
    files = _files(args.inputs)
    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    models: set[str] = set()
    seeds: set[int] = set()
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported calibration artifact: {path}")
        models.add(str(payload["model"]))
        seeds.add(int(payload["seed"]))
        for record in payload["records"]:
            value = torch.as_tensor(record["feature"], dtype=torch.float32).reshape(-1)
            magnitude = torch.linalg.vector_norm(value)
            value = torch.cat(
                (
                    value / magnitude.clamp_min(1e-12),
                    torch.log1p(magnitude).reshape(1),
                )
            )
            features.append(value.numpy())
            harmful = record["role"] == "attacker" and record["attack"] != "none"
            labels.append(int(harmful))
            metadata.append(
                {
                    "source": str(path),
                    "client_id": int(record["client_id"]),
                    "role": str(record["role"]),
                    "attack": str(record["attack"]),
                    "round": int(record["round"]),
                    "label": int(harmful),
                }
            )
    if len(models) != 1:
        raise ValueError(f"Probe inputs must use one model, found {sorted(models)}")
    if len(set(labels)) != 2:
        raise ValueError("Probe calibration requires both benign and harmful deltas")
    matrix = np.stack(features)
    target = np.asarray(labels, dtype=np.int64)
    estimator = LogisticRegression(
        C=float(args.c), class_weight="balanced", max_iter=2000, random_state=0
    )
    estimator.fit(matrix, target)
    probability = estimator.predict_proba(matrix)[:, 1]
    prediction = (probability >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, prediction, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(target, probability)),
        "examples": int(len(target)),
        "harmful_examples": int(target.sum()),
    }
    artifact = {
        "schema_version": 1,
        "method": "Safe-FedLLM Shadow-Level terminal-delta adaptation",
        "model": next(iter(models)),
        "feature_policy": "first_transformer_layer_lora_B_l2_plus_log_norm",
        "feature_dim": int(matrix.shape[1]),
        "coefficient": torch.from_numpy(estimator.coef_[0].astype(np.float32)),
        "intercept": float(estimator.intercept_[0]),
        "calibration": {
            "seeds": sorted(seeds),
            "metrics_in_sample": metrics,
            "files": [{"path": str(path), "sha256": _sha256(path)} for path in files],
            "records": metadata,
            "formal_seed_excluded": True,
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    report = destination.with_suffix(".json")
    report.write_text(
        json.dumps(
            {
                "artifact": str(destination),
                "model": artifact["model"],
                "feature_dim": artifact["feature_dim"],
                "calibration_seeds": sorted(seeds),
                "metrics_in_sample": metrics,
                "source_files": artifact["calibration"]["files"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
