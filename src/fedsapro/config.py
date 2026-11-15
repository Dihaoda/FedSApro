from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


PAPER_MODEL_KEYS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")
MODEL_KEYS = ("qwen2.5-0.5b", *PAPER_MODEL_KEYS)
ATTACKS = ("none", "label_flip", "textual_backdoor")
METHODS = ("fedavg", "fedprox", "scaffold", "indexguard", "flame", "fltrust")


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {source}")
    return value


def dump_yaml(value: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(value), handle, sort_keys=False, allow_unicode=True)


def deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get_path(config: Mapping[str, Any], dotted: str) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Missing configuration key: {dotted}")
        current = current[part]
    return current


def set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = config
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted}: {part} is not a mapping")
        current = child
    current[parts[-1]] = value


def resolve_config(
    path: str | Path,
    *,
    model: str | None = None,
    attack: str | None = None,
    method: str | None = None,
    seed: int | None = None,
    smoke: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_yaml(path)
    if model is not None:
        set_path(config, "model.key", model)
    if attack is not None:
        set_path(config, "attack.name", attack)
    if method is not None:
        set_path(config, "method.name", method)
    if seed is not None:
        set_path(config, "experiment.seed", int(seed))

    if smoke:
        config = deep_update(
            config,
            {
                "experiment": {"data_root": "data/smoke_v1", "output_root": "results/smoke"},
                "data": {
                    "num_clients": 3,
                    "standard_clients": 1,
                    "heterogeneous_clients": 1,
                    "attacker_clients": 1,
                    "samples_per_client": 20,
                    "test_samples_per_domain": 24,
                    "root_samples": 12,
                },
                "federated": {
                    "rounds": 1,
                    "clients_per_round": 3,
                    "batch_size": 2,
                    "micro_batch_size": 2,
                    "eval_batch_size": 2,
                    "max_length": 64,
                    "eval_every": 1,
                },
                "reporting": {"run_class": "smoke_test"},
            },
        )

    for dotted, value in (overrides or {}).items():
        set_path(config, dotted, value)

    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    model = str(get_path(config, "model.key"))
    task = str(get_path(config, "model.task"))
    classifier_initialization = str(get_path(config, "model.classifier_initialization"))
    attack = str(get_path(config, "attack.name"))
    method = str(get_path(config, "method.name"))
    if model not in MODEL_KEYS:
        raise ValueError(f"Unknown model {model!r}; choose from {MODEL_KEYS}")
    if task != "sequence_classification":
        raise ValueError("FedSApro requires model.task=sequence_classification")
    if classifier_initialization not in {"random", "label_token_lm_head"}:
        raise ValueError(
            "model.classifier_initialization must be random or label_token_lm_head"
        )
    if attack not in ATTACKS:
        raise ValueError(f"Unknown attack {attack!r}; choose from {ATTACKS}")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")

    roles = sum(
        int(get_path(config, f"data.{key}"))
        for key in ("standard_clients", "heterogeneous_clients", "attacker_clients")
    )
    total = int(get_path(config, "data.num_clients"))
    if roles != total:
        raise ValueError(f"Client role counts sum to {roles}, expected {total}")
    participating = int(get_path(config, "federated.clients_per_round"))
    if participating != total:
        raise ValueError("The paper configuration requires full client participation each round")
    samples = int(get_path(config, "data.samples_per_client"))
    if samples < 2 or samples % 2:
        raise ValueError("samples_per_client must be an even integer >= 2")
    hetero_amazon = samples * float(get_path(config, "data.heterogeneous_amazon_fraction"))
    if round(hetero_amazon) != hetero_amazon or int(hetero_amazon) % 2:
        raise ValueError("The heterogeneous Amazon sample count must be an even integer")
    if int(get_path(config, "federated.rounds")) < 1:
        raise ValueError("rounds must be positive")
    if float(get_path(config, "federated.server_learning_rate")) <= 0:
        raise ValueError("server_learning_rate must be positive")
    if float(get_path(config, "attack.update_scale")) <= 0:
        raise ValueError("attack.update_scale must be positive")
    effective_batch = int(get_path(config, "federated.batch_size"))
    micro_batch = int(get_path(config, "federated.micro_batch_size"))
    if micro_batch < 1 or effective_batch < micro_batch or effective_batch % micro_batch:
        raise ValueError("batch_size must be a positive multiple of micro_batch_size")


def config_digest(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_run_dir(config: Mapping[str, Any]) -> Path:
    root = Path(str(get_path(config, "experiment.output_root")))
    model = str(get_path(config, "model.key"))
    attack = str(get_path(config, "attack.name"))
    method = str(get_path(config, "method.name"))
    seed = int(get_path(config, "experiment.seed"))
    return root / model / attack / method / f"seed_{seed}"

