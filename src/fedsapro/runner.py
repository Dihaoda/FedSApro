from __future__ import annotations

import csv
import copy
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from .aggregators import ClientUpdate, aggregate
from .config import config_digest, dump_yaml, experiment_run_dir, get_path
from .data import (
    ATTACKER,
    apply_attack,
    load_client_rows,
    role_map,
    triggered_evaluation_rows,
    validate_data,
)
from .diagnostics import summarize_run
from .modeling import evaluate_accuracy, evaluate_target_rate, load_model
from .state import (
    build_layout,
    gather_first_lora_b,
    pack_parameters,
    trainable_named_parameters,
    unpack_parameters,
)
from .training import train_local
from .utils import atomic_json_dump, derive_seed, read_jsonl, seed_everything


LOGGER = logging.getLogger(__name__)


CLIENT_FIELDS = [
    "round",
    "client_id",
    "role",
    "attack",
    "sample_count",
    "steps",
    "mean_loss",
    "peak_memory_mb",
    "update_norm",
    "raw_coefficient",
    "effective_coefficient",
    "selected",
]
ROUND_FIELDS = [
    "round",
    "standard_accuracy",
    "heterogeneous_accuracy",
    "domain_gap",
    "backdoor_asr_standard",
    "backdoor_asr_heterogeneous",
    "round_seconds",
]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_csv_header(path: Path, fields: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def _append_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _configure_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def _checkpoint_payload(
    *,
    config: Mapping[str, Any],
    round_number: int,
    global_vector: torch.Tensor,
    scaffold_server: torch.Tensor | None,
    scaffold_clients: dict[int, torch.Tensor] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_digest": config_digest(config),
        "round": round_number,
        "global_vector": global_vector.half().cpu(),
        "scaffold_server": scaffold_server.half().cpu() if scaffold_server is not None else None,
        "scaffold_clients": {
            client_id: vector.half().cpu() for client_id, vector in (scaffold_clients or {}).items()
        },
    }


def _load_checkpoint(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("config_digest") != config_digest(config):
        raise RuntimeError("Checkpoint configuration digest does not match the requested run")
    return checkpoint


def run_experiment(
    config: Mapping[str, Any], *, resume: bool = False, force: bool = False
) -> dict[str, Any]:
    validate_data(config)
    run_dir = experiment_run_dir(config)
    if run_dir.exists() and force:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir)
    dump_yaml(config, run_dir / "resolved_config.yaml")
    provenance = {
        "git_commit": _git_commit(),
        "github_commit": os.environ.get("FEDSA_GITHUB_COMMIT"),
        "config_digest": config_digest(config),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "run_class": get_path(config, "reporting.run_class"),
    }
    atomic_json_dump(provenance, run_dir / "provenance.json")
    _write_csv_header(run_dir / "client_rounds.csv", CLIENT_FIELDS)
    _write_csv_header(run_dir / "round_metrics.csv", ROUND_FIELDS)

    if not torch.cuda.is_available():
        raise RuntimeError("The paper experiment requires a CUDA GPU")
    device = torch.device("cuda:0")
    seed = int(get_path(config, "experiment.seed"))
    seed_everything(seed)
    model, tokenizer = load_model(config, device)
    named_parameters = trainable_named_parameters(model)
    layout = build_layout(named_parameters)
    global_vector = pack_parameters(named_parameters, dtype=torch.float32, device="cpu")
    method = str(get_path(config, "method.name"))
    attack_name = str(get_path(config, "attack.name"))
    roles = role_map(config)
    client_rows: dict[int, list[dict[str, Any]]] = {}
    for client_id, role in roles.items():
        rows = load_client_rows(config, client_id)
        if role == ATTACKER:
            rows = apply_attack(
                rows,
                attack=attack_name,
                config=config,
                seed=derive_seed(seed, 40, client_id),
            )
        client_rows[client_id] = rows

    scaffold_server: torch.Tensor | None = None
    scaffold_clients: dict[int, torch.Tensor] | None = None
    if method == "scaffold":
        scaffold_server = torch.zeros_like(global_vector)
        scaffold_clients = {client_id: torch.zeros_like(global_vector) for client_id in roles}

    checkpoint_path = run_dir / "checkpoint_latest.pt"
    start_round = 1
    if resume and checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path, config)
        start_round = int(checkpoint["round"]) + 1
        global_vector = checkpoint["global_vector"].float()
        if method == "scaffold":
            scaffold_server = checkpoint["scaffold_server"].float()
            scaffold_clients = {
                int(client_id): value.float()
                for client_id, value in checkpoint["scaffold_clients"].items()
            }
        LOGGER.info("Resuming from round %d", start_round)
    elif resume:
        LOGGER.info("No checkpoint found; starting a new run")
    elif checkpoint_path.exists() and not force:
        raise RuntimeError(f"Run directory already contains a checkpoint: {run_dir}; use --resume or --force")

    data_root = Path(str(get_path(config, "experiment.data_root")))
    amazon_test = read_jsonl(data_root / "test_amazon.jsonl")
    imdb_test = read_jsonl(data_root / "test_imdb.jsonl")
    backdoor_target = int(get_path(config, "attack.backdoor_target"))
    trigger = str(get_path(config, "data.trigger"))
    trigger_position = str(config["data"].get("trigger_position", "prefix"))
    triggered_amazon_test = triggered_evaluation_rows(
        amazon_test, trigger=trigger, target=backdoor_target, position=trigger_position
    )
    triggered_imdb_test = triggered_evaluation_rows(
        imdb_test, trigger=trigger, target=backdoor_target, position=trigger_position
    )
    root_rows = read_jsonl(data_root / "server_root.jsonl")
    rounds = int(get_path(config, "federated.rounds"))
    eval_every = int(get_path(config, "federated.eval_every"))
    checkpoint_every = int(get_path(config, "federated.checkpoint_every"))
    labels = list(get_path(config, "data.labels"))
    max_length = int(get_path(config, "federated.max_length"))
    eval_batch_size = max(1, int(get_path(config, "federated.eval_batch_size")))
    default_server_learning_rate = float(
        get_path(config, "federated.server_learning_rate")
    )
    method_server_learning_rates = config["method"].get("server_learning_rates", {})
    server_learning_rate = float(
        method_server_learning_rates.get(method, default_server_learning_rate)
    )
    attack_scales = get_path(config, "attack.update_scales")
    attack_update_scale = float(
        attack_scales.get(attack_name, get_path(config, "attack.update_scale"))
    )

    for round_number in range(start_round, rounds + 1):
        started = time.monotonic()
        LOGGER.info("Starting round %d/%d", round_number, rounds)
        updates: list[ClientUpdate] = []
        local_metadata: dict[int, dict[str, Any]] = {}
        control_deltas: list[torch.Tensor] = []
        cache_root = str(config.get("calibration", {}).get("shared_update_cache_dir", "")).strip()
        cache_path: Path | None = None
        cache_attack = copy.deepcopy(config["attack"])
        cache_reference_scales = config.get("calibration", {}).get(
            "shared_update_cache_reference_scales", {}
        )
        cache_reference_scale = float(
            cache_reference_scales.get(attack_name, attack_update_scale)
        )
        cache_attack.setdefault("update_scales", {}).update(
            {name: float(value) for name, value in cache_reference_scales.items()}
        )
        cache_signature = config_digest(
            {
                "model": config["model"],
                "data": config["data"],
                "federated": config["federated"],
                "attack": cache_attack,
                "seed": seed,
                "round": round_number,
            }
        )
        if cache_root:
            if rounds != 1 or method in {"fedprox", "scaffold"}:
                raise ValueError(
                    "calibration.shared_update_cache_dir only supports one-round methods "
                    "with method-independent local training"
                )
            cache_path = (
                Path(cache_root)
                / str(get_path(config, "model.key"))
                / attack_name
                / f"seed_{seed}_round_{round_number}.pt"
            )
        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("schema_version") != 1 or cached.get("signature") != cache_signature:
                raise RuntimeError(f"Shared client-update cache mismatch: {cache_path}")
            updates = [
                ClientUpdate(
                    int(record["client_id"]),
                    str(record["role"]),
                    torch.as_tensor(record["delta"], dtype=torch.float32),
                    int(record["sample_count"]),
                )
                for record in cached["updates"]
            ]
            local_metadata = {int(key): value for key, value in cached["local_metadata"].items()}
            LOGGER.info("Loaded %d shared client updates from %s", len(updates), cache_path)
        else:
            if not math.isclose(attack_update_scale, cache_reference_scale):
                raise RuntimeError(
                    "The shared cache is missing and the requested attack scale differs "
                    "from calibration.shared_update_cache_reference_scales"
                )
            for client_id, role in roles.items():
                result = train_local(
                    model,
                    tokenizer,
                    client_rows[client_id],
                    config=config,
                    device=device,
                    global_vector=global_vector,
                    seed=derive_seed(seed, round_number, client_id),
                    method=method,
                    client_control=(scaffold_clients or {}).get(client_id),
                    server_control=scaffold_server,
                )
                if method == "scaffold":
                    assert scaffold_clients is not None and result.new_control is not None
                    scaffold_clients[client_id] = result.new_control
                    assert result.control_delta is not None
                    control_deltas.append(result.control_delta)
                submitted_delta = result.delta
                if role == ATTACKER and attack_name != "none":
                    scale_scope = str(config["attack"].get("scale_scope", "all"))
                    if scale_scope == "all":
                        submitted_delta = submitted_delta * attack_update_scale
                    elif scale_scope == "lora_only":
                        submitted_delta = submitted_delta.clone()
                        for offset, count in layout.qv_slices:
                            submitted_delta[offset : offset + count].mul_(attack_update_scale)
                    else:
                        raise ValueError(f"Unknown attack.scale_scope: {scale_scope}")
                updates.append(
                    ClientUpdate(client_id, role, submitted_delta, len(client_rows[client_id]))
                )
                local_metadata[client_id] = {
                    "steps": result.steps,
                    "mean_loss": result.mean_loss,
                    "peak_memory_mb": result.peak_memory_mb,
                    "update_norm": float(torch.linalg.vector_norm(submitted_delta).item()),
                }
                LOGGER.info(
                    "round=%d client=%d role=%s steps=%d loss=%.6f norm=%.6f peak_mb=%.1f",
                    round_number,
                    client_id,
                    role,
                    result.steps,
                    result.mean_loss,
                    local_metadata[client_id]["update_norm"],
                    local_metadata[client_id]["peak_memory_mb"],
                )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                torch.save(
                    {
                        "schema_version": 1,
                        "signature": cache_signature,
                        "updates": [
                            {
                                "client_id": update.client_id,
                                "role": update.role,
                                "delta": update.delta.float().cpu(),
                                "sample_count": update.sample_count,
                            }
                            for update in updates
                        ],
                        "local_metadata": local_metadata,
                    },
                    temporary,
                )
                temporary.replace(cache_path)
                LOGGER.info("Saved shared client updates to %s", cache_path)

        cache_scale_ratio = attack_update_scale / max(cache_reference_scale, 1e-12)
        if cache_path is not None and cache_path.exists() and not math.isclose(
            cache_scale_ratio, 1.0
        ):
            rescaled_updates: list[ClientUpdate] = []
            for update in updates:
                submitted_delta = update.delta
                if update.role == ATTACKER and attack_name != "none":
                    submitted_delta = submitted_delta.clone()
                    scale_scope = str(config["attack"].get("scale_scope", "all"))
                    if scale_scope == "all":
                        submitted_delta.mul_(cache_scale_ratio)
                    elif scale_scope == "lora_only":
                        for offset, count in layout.qv_slices:
                            submitted_delta[offset : offset + count].mul_(cache_scale_ratio)
                    else:
                        raise ValueError(f"Unknown attack.scale_scope: {scale_scope}")
                rescaled_updates.append(
                    ClientUpdate(
                        update.client_id,
                        update.role,
                        submitted_delta,
                        update.sample_count,
                    )
                )
                local_metadata[update.client_id]["update_norm"] = float(
                    torch.linalg.vector_norm(submitted_delta).item()
                )
            updates = rescaled_updates
            LOGGER.info(
                "Rescaled cached %s attacker updates from %.6g to %.6g",
                attack_name,
                cache_reference_scale,
                attack_update_scale,
            )

        if method == "scaffold":
            assert scaffold_server is not None
            scaffold_server = scaffold_server + torch.stack(control_deltas).sum(dim=0) / len(roles)

        if bool(config.get("calibration", {}).get("capture_safefedllm_features", False)):
            feature_records = []
            for update in updates:
                feature_records.append(
                    {
                        "client_id": update.client_id,
                        "role": update.role,
                        "attack": attack_name if update.role == ATTACKER else "none",
                        "round": round_number,
                        "feature": gather_first_lora_b(update.delta, layout).half().cpu(),
                    }
                )
            torch.save(
                {
                    "schema_version": 1,
                    "model": str(get_path(config, "model.key")),
                    "seed": seed,
                    "config_digest": config_digest(config),
                    "records": feature_records,
                },
                run_dir / f"safefedllm_features_round_{round_number:03d}.pt",
            )

        root_delta = None
        if method == "fltrust":
            root_result = train_local(
                model,
                tokenizer,
                root_rows,
                config=config,
                device=device,
                global_vector=global_vector,
                seed=derive_seed(seed, round_number, 90_000),
                method="fedavg",
                epochs_override=int(get_path(config, "method.fltrust.root_epochs")),
            )
            root_delta = root_result.delta

        generator = torch.Generator().manual_seed(derive_seed(seed, round_number, 80_000))
        aggregation = aggregate(
            method,
            updates,
            layout=layout,
            method_config=config["method"],
            root_delta=root_delta,
            generator=generator,
        )
        global_vector = global_vector + server_learning_rate * aggregation.delta
        unpack_parameters(named_parameters, global_vector)

        client_log_rows: list[dict[str, Any]] = []
        for update in updates:
            metadata = local_metadata[update.client_id]
            client_log_rows.append(
                {
                    "round": round_number,
                    "client_id": update.client_id,
                    "role": update.role,
                    "attack": attack_name if update.role == ATTACKER else "none",
                    "sample_count": update.sample_count,
                    "steps": metadata["steps"],
                    "mean_loss": f"{metadata['mean_loss']:.10g}",
                    "peak_memory_mb": f"{metadata['peak_memory_mb']:.6f}",
                    "update_norm": f"{metadata['update_norm']:.10g}",
                    "raw_coefficient": f"{aggregation.raw_coefficients[update.client_id]:.10g}",
                    "effective_coefficient": f"{aggregation.effective_coefficients[update.client_id]:.10g}",
                    "selected": int(aggregation.selected[update.client_id]),
                }
            )
        _append_csv(run_dir / "client_rounds.csv", CLIENT_FIELDS, client_log_rows)
        _append_jsonl(
            run_dir / "aggregation_diagnostics.jsonl",
            {"round": round_number, "diagnostics": aggregation.diagnostics},
        )

        do_evaluate = round_number == 1 or round_number == rounds or round_number % eval_every == 0
        if do_evaluate:
            standard_accuracy = evaluate_accuracy(
                model,
                tokenizer,
                amazon_test,
                label_names=labels,
                max_length=max_length,
                batch_size=eval_batch_size,
                device=device,
            )
            heterogeneous_accuracy = evaluate_accuracy(
                model,
                tokenizer,
                imdb_test,
                label_names=labels,
                max_length=max_length,
                batch_size=eval_batch_size,
                device=device,
            )
            gap = standard_accuracy - heterogeneous_accuracy
            if attack_name == "textual_backdoor":
                backdoor_asr_standard = evaluate_target_rate(
                    model,
                    tokenizer,
                    triggered_amazon_test,
                    target=backdoor_target,
                    label_names=labels,
                    max_length=max_length,
                    batch_size=eval_batch_size,
                    device=device,
                )
                backdoor_asr_heterogeneous = evaluate_target_rate(
                    model,
                    tokenizer,
                    triggered_imdb_test,
                    target=backdoor_target,
                    label_names=labels,
                    max_length=max_length,
                    batch_size=eval_batch_size,
                    device=device,
                )
            else:
                backdoor_asr_standard = None
                backdoor_asr_heterogeneous = None
            LOGGER.info(
                "round=%d standard_acc=%.4f heterogeneous_acc=%.4f gap=%.4f "
                "backdoor_asr_standard=%s backdoor_asr_heterogeneous=%s",
                round_number,
                standard_accuracy,
                heterogeneous_accuracy,
                gap,
                "n/a" if backdoor_asr_standard is None else f"{backdoor_asr_standard:.4f}",
                (
                    "n/a"
                    if backdoor_asr_heterogeneous is None
                    else f"{backdoor_asr_heterogeneous:.4f}"
                ),
            )
            round_row = {
                "round": round_number,
                "standard_accuracy": f"{standard_accuracy:.10g}",
                "heterogeneous_accuracy": f"{heterogeneous_accuracy:.10g}",
                "domain_gap": f"{gap:.10g}",
                "backdoor_asr_standard": (
                    "" if backdoor_asr_standard is None else f"{backdoor_asr_standard:.10g}"
                ),
                "backdoor_asr_heterogeneous": (
                    ""
                    if backdoor_asr_heterogeneous is None
                    else f"{backdoor_asr_heterogeneous:.10g}"
                ),
                "round_seconds": f"{time.monotonic() - started:.6f}",
            }
        else:
            round_row = {
                "round": round_number,
                "standard_accuracy": "",
                "heterogeneous_accuracy": "",
                "domain_gap": "",
                "backdoor_asr_standard": "",
                "backdoor_asr_heterogeneous": "",
                "round_seconds": f"{time.monotonic() - started:.6f}",
            }
        _append_csv(run_dir / "round_metrics.csv", ROUND_FIELDS, [round_row])

        if round_number % checkpoint_every == 0 or round_number == rounds:
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(
                _checkpoint_payload(
                    config=config,
                    round_number=round_number,
                    global_vector=global_vector,
                    scaffold_server=scaffold_server,
                    scaffold_clients=scaffold_clients,
                ),
                temporary,
            )
            os.replace(temporary, checkpoint_path)
        LOGGER.info("Completed round %d in %.1f seconds", round_number, time.monotonic() - started)

    unpack_parameters(named_parameters, global_vector)
    model.save_pretrained(run_dir / "adapter_final", safe_serialization=True)
    tokenizer.save_pretrained(run_dir / "adapter_final")
    summary = summarize_run(config, run_dir)
    LOGGER.info("Run completed: accepted=%s summary=%s", summary["accepted"], run_dir / "summary.json")
    return summary

