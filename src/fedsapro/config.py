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
PAPER_METHODS = ("fedavg", "fedprox", "scaffold", "indexguard", "flame", "fltrust")
METHODS = (
    "fedavg",
    "fedprox",
    "scaffold",
    "indexguard",
    "flame",
    "fltrust",
    "safefedllm",
    "fedgraph",
)


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
    if str(config["data"].get("test_profile", "random")) not in {
        "random",
        "random_length_pools",
        "length_extremes",
        "lexical_length_extremes",
    }:
        raise ValueError(
            "data.test_profile must be random, random_length_pools, length_extremes, "
            "or lexical_length_extremes"
        )

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
    for role_name, fraction in (
        ("heterogeneous", float(get_path(config, "data.heterogeneous_amazon_fraction"))),
        ("attacker", float(config["data"].get("attacker_amazon_fraction", 1.0))),
    ):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"The {role_name} Amazon fraction must be in [0, 1]")
        amazon_samples = samples * fraction
        if round(amazon_samples) != amazon_samples or int(amazon_samples) % 2:
            raise ValueError(f"The {role_name} Amazon sample count must be an even integer")
    if int(get_path(config, "federated.rounds")) < 1:
        raise ValueError("rounds must be positive")
    if float(get_path(config, "federated.server_learning_rate")) <= 0:
        raise ValueError("server_learning_rate must be positive")
    for method_name, value in config["method"].get("server_learning_rates", {}).items():
        valid_method = method_name in METHODS
        if isinstance(value, Mapping):
            valid_keys = set(ATTACKS) | {"default"}
            valid_value = bool(value) and set(value) <= valid_keys and all(
                float(item) > 0 for item in value.values()
            )
        else:
            valid_value = float(value) > 0
        if not valid_method or not valid_value:
            raise ValueError(
                f"method.server_learning_rates.{method_name} must name a known method "
                "and contain positive scalar or per-attack values"
            )
    if float(get_path(config, "attack.update_scale")) <= 0:
        raise ValueError("attack.update_scale must be positive")
    if str(config["attack"].get("scale_scope", "all")) not in {"all", "lora_only"}:
        raise ValueError("attack.scale_scope must be all or lora_only")
    scaffold_control_scale = float(config["method"].get("scaffold_control_scale", 1.0))
    if not 0.0 < scaffold_control_scale <= 1.0:
        raise ValueError("method.scaffold_control_scale must be in (0, 1]")
    update_scales = get_path(config, "attack.update_scales")
    for attack_name in ATTACKS:
        if float(update_scales[attack_name]) <= 0:
            raise ValueError(f"attack.update_scales.{attack_name} must be positive")
    for attack_name, value in config["attack"].get("norm_ratio_targets", {}).items():
        if attack_name not in ATTACKS or attack_name == "none" or float(value) <= 0:
            raise ValueError(
                f"attack.norm_ratio_targets.{attack_name} must name an attacked "
                "condition and be positive"
            )
    if method == "safefedllm":
        safe_config = get_path(config, "method.safefedllm")
        if float(safe_config.get("eta", 7.0)) <= 0:
            raise ValueError("method.safefedllm.eta must be positive")
        min_weight = float(safe_config.get("min_weight", 0.0))
        if not 0.0 <= min_weight < 1.0:
            raise ValueError("method.safefedllm.min_weight must be in [0, 1)")
        norm_guard_fraction = float(safe_config.get("norm_guard_fraction", 0.0))
        if not 0.0 <= norm_guard_fraction < 1.0:
            raise ValueError("method.safefedllm.norm_guard_fraction must be in [0, 1)")
    if method == "fedgraph":
        graph_config = get_path(config, "method.fedgraph")
        threshold = float(graph_config.get("similarity_threshold", 0.8))
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("method.fedgraph.similarity_threshold must be in [-1, 1]")
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

