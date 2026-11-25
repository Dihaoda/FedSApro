from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .aggregators import AggregationResult, ClientUpdate
from .modeling import build_prompt
from .state import ParameterLayout, unpack_parameters


@dataclass
class _AnchorObservation:
    representation: torch.Tensor
    loss: float


def _balanced_anchors(
    rows: Sequence[Mapping[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Take a deterministic, approximately class-balanced server anchor set."""
    by_label: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(int(row["label"]), []).append(dict(row))
    labels = sorted(by_label)
    if not labels:
        raise ValueError("HARP-FedLLM requires non-empty server anchors")
    selected: list[dict[str, Any]] = []
    offsets = {label: 0 for label in labels}
    while len(selected) < min(count, len(rows)):
        progressed = False
        for label in labels:
            offset = offsets[label]
            if offset < len(by_label[label]) and len(selected) < count:
                selected.append(by_label[label][offset])
                offsets[label] += 1
                progressed = True
        if not progressed:
            break
    return selected


def _representative_layer_ids(model: torch.nn.Module) -> tuple[int, int, int]:
    layers = int(getattr(model.config, "num_hidden_layers"))
    # hidden_states[0] is the embedding output; the remaining entries are
    # transformer-layer outputs.
    return max(1, layers // 3), max(1, (2 * layers) // 3), layers


def _encoded_batches(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    label_names: Sequence[str],
    max_length: int,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoded = tokenizer(
            [build_prompt(str(row["text"]), label_names) for row in chunk],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [int(row["label"]) for row in chunk], dtype=torch.long
        )
        batches.append(dict(encoded))
    return batches


@torch.inference_mode()
def _observe(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    layer_ids: Sequence[int],
    num_labels: int,
    device: torch.device,
) -> _AnchorObservation:
    model.eval()
    sums: dict[tuple[int, int], torch.Tensor] = {}
    counts: dict[int, int] = {label: 0 for label in range(num_labels)}
    total_loss = 0.0
    total_count = 0
    mixed_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.dtype in {torch.bfloat16, torch.float16}
    }
    amp_dtype = next(iter(mixed_dtypes), torch.bfloat16)
    use_amp = device.type == "cuda" and bool(mixed_dtypes)
    for cpu_batch in batches:
        batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
        labels = batch["labels"]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(**batch, output_hidden_states=True, return_dict=True)
        logits = output.logits.float()
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
        total_count += int(labels.numel())
        last_positions = batch["attention_mask"].sum(dim=1) - 1
        row_ids = torch.arange(labels.numel(), device=device)
        for layer_id in layer_ids:
            hidden = output.hidden_states[layer_id][row_ids, last_positions].float()
            for label in range(num_labels):
                mask = labels == label
                if not bool(mask.any()):
                    continue
                value = hidden[mask].sum(dim=0).cpu()
                key = (layer_id, label)
                sums[key] = value if key not in sums else sums[key] + value
        for label in range(num_labels):
            counts[label] += int((labels == label).sum().item())
        del output
    means: list[torch.Tensor] = []
    for layer_id in layer_ids:
        for label in range(num_labels):
            if counts[label] <= 0:
                raise ValueError(f"No HARP-FedLLM anchor has label {label}")
            means.append(sums[(layer_id, label)] / counts[label])
    return _AnchorObservation(
        representation=torch.stack(means),
        loss=total_loss / max(1, total_count),
    )


def _parameter_slices(layout: ParameterLayout) -> list[tuple[str, int, int]]:
    values: list[tuple[str, int, int]] = []
    offset = 0
    for name, count in zip(layout.names, layout.numels):
        values.append((name, offset, count))
        offset += count
    return values


def _update_fingerprint(
    vector: torch.Tensor,
    layout: ParameterLayout,
    layer_ids: Sequence[int],
) -> tuple[torch.Tensor, list[float]]:
    slices = _parameter_slices(layout)
    selected: list[torch.Tensor] = []
    log_rms: list[float] = []
    for layer_id in layer_ids:
        pieces: list[torch.Tensor] = []
        marker = re.compile(rf"\.layers\.{int(layer_id) - 1}\.")
        for name, offset, count in slices:
            if "lora_" in name.lower() and marker.search(name.lower()):
                pieces.append(vector[offset : offset + count].float().cpu())
        if not pieces:
            raise RuntimeError(f"No LoRA update found for transformer layer {layer_id - 1}")
        block = torch.cat(pieces)
        selected.append(block)
        rms = float(torch.linalg.vector_norm(block).item()) / math.sqrt(block.numel())
        log_rms.append(math.log(max(rms, 1e-12)))
    joined = torch.cat(selected)
    joined = joined / torch.linalg.vector_norm(joined).clamp_min(1e-12)
    return joined, log_rms


def _robust_upper_excess(values: Sequence[float], threshold: float) -> tuple[np.ndarray, dict[str, float]]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        q25, q75 = np.quantile(array, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if scale < 1e-8:
        scale = max(float(array.std()), 1e-8)
    z = (array - median) / scale
    return np.maximum(0.0, z - float(threshold)), {
        "median": median,
        "robust_scale": scale,
        "threshold": float(threshold),
    }


def _mode_analysis(matrix: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    components = max(2, min(int(config.get("pca_components", 6)), len(matrix) - 1, matrix.shape[1]))
    coordinates = PCA(n_components=components, random_state=0).fit_transform(matrix)
    coordinates = StandardScaler().fit_transform(coordinates)
    labels = HDBSCAN(
        min_cluster_size=max(2, int(config.get("min_mode_size", 4))),
        min_samples=1,
        allow_single_cluster=True,
    ).fit_predict(coordinates)
    groups = {
        int(label): int((labels == label).sum())
        for label in np.unique(labels)
        if int(label) >= 0
    }
    if groups:
        largest = sorted(groups, key=lambda label: (-groups[label], label))[0]
        center = coordinates[labels == largest].mean(axis=0)
    else:
        center = np.median(coordinates, axis=0)
    novelty = np.linalg.norm(coordinates - center, axis=1)
    unsupported = np.asarray(
        [label < 0 or groups.get(int(label), 0) < int(config.get("min_mode_size", 4)) for label in labels],
        dtype=bool,
    )
    return labels, novelty, unsupported


def aggregate_harp_fedllm(
    model: torch.nn.Module,
    tokenizer: Any,
    updates: Sequence[ClientUpdate],
    *,
    global_vector: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    layout: ParameterLayout,
    anchor_rows: Sequence[Mapping[str, Any]],
    label_names: Sequence[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    config: Mapping[str, Any],
) -> AggregationResult:
    """Open-set, risk-only soft aggregation for federated LLM fine-tuning.

    The representation-space mode geometry is used to report novelty. Novelty
    alone is deliberately excluded from the risk score, because an unseen
    benign domain can be far from the server's standard-domain mode.
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty update list")
    anchors = _balanced_anchors(anchor_rows, int(config.get("anchor_samples", 48)))
    suffix = str(config.get("perturbation_suffix", " Additional context is unavailable."))
    perturbed = [dict(row, text=str(row["text"]) + suffix) for row in anchors]
    layer_ids = _representative_layer_ids(model)
    clean_batches = _encoded_batches(
        tokenizer,
        anchors,
        label_names=label_names,
        max_length=max_length,
        batch_size=max(1, batch_size),
    )
    perturbed_batches = _encoded_batches(
        tokenizer,
        perturbed,
        label_names=label_names,
        max_length=max_length,
        batch_size=max(1, batch_size),
    )

    unpack_parameters(named_parameters, global_vector)
    baseline_clean = _observe(
        model,
        clean_batches,
        layer_ids=layer_ids,
        num_labels=len(label_names),
        device=device,
    )
    baseline_perturbed = _observe(
        model,
        perturbed_batches,
        layer_ids=layer_ids,
        num_labels=len(label_names),
        device=device,
    )

    update_blocks: list[torch.Tensor] = []
    representation_blocks: list[torch.Tensor] = []
    log_norms: list[float] = []
    anchor_harms: list[float] = []
    sensitivities: list[float] = []
    layer_log_rms: list[list[float]] = []
    for update in updates:
        update_block, layer_rms = _update_fingerprint(update.delta, layout, layer_ids)
        update_blocks.append(update_block)
        layer_log_rms.append(layer_rms)
        log_norms.append(math.log(max(float(torch.linalg.vector_norm(update.delta.float()).item()), 1e-12)))

        unpack_parameters(named_parameters, global_vector + update.delta.float().cpu())
        clean = _observe(
            model,
            clean_batches,
            layer_ids=layer_ids,
            num_labels=len(label_names),
            device=device,
        )
        noisy = _observe(
            model,
            perturbed_batches,
            layer_ids=layer_ids,
            num_labels=len(label_names),
            device=device,
        )
        clean_shift = clean.representation - baseline_clean.representation
        response_shift = (noisy.representation - clean.representation) - (
            baseline_perturbed.representation - baseline_clean.representation
        )
        block = torch.cat((clean_shift.reshape(-1), response_shift.reshape(-1)))
        representation_blocks.append(block / torch.linalg.vector_norm(block).clamp_min(1e-12))
        anchor_harms.append(clean.loss - baseline_clean.loss)
        sensitivities.append(float(response_shift.square().mean().sqrt().item()))

    # Give update direction, representation direction, and per-layer magnitude
    # equal block-level influence in the discovery geometry.
    update_matrix = torch.stack(update_blocks).numpy()
    representation_matrix = torch.stack(representation_blocks).numpy()
    rms_matrix = np.asarray(layer_log_rms, dtype=np.float32)
    rms_scale = np.std(rms_matrix, axis=0, keepdims=True)
    rms_matrix = (rms_matrix - np.median(rms_matrix, axis=0, keepdims=True)) / np.maximum(rms_scale, 1e-6)
    mode_matrix = np.concatenate(
        (
            update_matrix / math.sqrt(3.0),
            representation_matrix / math.sqrt(3.0),
            rms_matrix / math.sqrt(3.0 * rms_matrix.shape[1]),
        ),
        axis=1,
    )
    labels, novelty, unsupported = _mode_analysis(mode_matrix, config)

    norm_excess, norm_reference = _robust_upper_excess(
        log_norms, float(config.get("norm_z_threshold", 3.0))
    )
    # Less improvement than the median client is not damage. Only an absolute
    # loss increase relative to the unadapted global model is one-sided harm.
    positive_anchor_harms = [max(0.0, value) for value in anchor_harms]
    harm_excess, harm_reference = _robust_upper_excess(
        positive_anchor_harms, float(config.get("harm_z_threshold", 2.5))
    )
    sensitivity_excess, sensitivity_reference = _robust_upper_excess(
        sensitivities, float(config.get("sensitivity_z_threshold", 3.0))
    )
    evidence = np.maximum.reduce((norm_excess, harm_excess, sensitivity_excess))
    # A small/unsupported mode is corroborating evidence, never an independent
    # rejection rule. This is the key heterogeneous-benign safeguard.
    mode_bonus = float(config.get("mode_bonus", 0.5)) * unsupported * np.minimum(1.0, evidence)
    risk = evidence + mode_bonus
    temperature = float(config.get("temperature", 1.0))
    minimum = float(config.get("minimum_factor", 0.01))
    factors = np.maximum(minimum, np.exp(-risk / temperature))
    threshold = float(config.get("selection_threshold", 0.25))

    raw = {update.client_id: float(update.sample_count) for update in updates}
    effective = {
        update.client_id: raw[update.client_id] * float(factor)
        for update, factor in zip(updates, factors)
    }
    total = sum(effective.values())
    if total <= 0:
        raise RuntimeError("HARP-FedLLM assigned zero total contribution")
    aggregate = torch.zeros_like(updates[0].delta, dtype=torch.float32, device="cpu")
    for update in updates:
        aggregate.add_(update.delta.float().cpu(), alpha=effective[update.client_id] / total)
    unpack_parameters(named_parameters, global_vector)

    ids = [update.client_id for update in updates]
    return AggregationResult(
        delta=aggregate,
        raw_coefficients=raw,
        effective_coefficients=effective,
        selected={client_id: bool(factor >= threshold) for client_id, factor in zip(ids, factors)},
        diagnostics={
            "variant": "open-set-risk-only-soft-weighting",
            "anchor_source": "standard-domain-server-root-only",
            "anchor_samples": len(anchors),
            "representative_layer_outputs": list(layer_ids),
            "perturbation_suffix": suffix,
            "client_ids": ids,
            "mode_labels": labels.tolist(),
            "domain_novelty_not_penalized": {client_id: float(value) for client_id, value in zip(ids, novelty)},
            "unsupported_mode": {client_id: bool(value) for client_id, value in zip(ids, unsupported)},
            "log_update_norm": {client_id: value for client_id, value in zip(ids, log_norms)},
            "anchor_harm": {client_id: value for client_id, value in zip(ids, anchor_harms)},
            "perturbation_sensitivity": {client_id: value for client_id, value in zip(ids, sensitivities)},
            "norm_excess": {client_id: float(value) for client_id, value in zip(ids, norm_excess)},
            "harm_excess": {client_id: float(value) for client_id, value in zip(ids, harm_excess)},
            "sensitivity_excess": {client_id: float(value) for client_id, value in zip(ids, sensitivity_excess)},
            "risk": {client_id: float(value) for client_id, value in zip(ids, risk)},
            "security_factor": {client_id: float(value) for client_id, value in zip(ids, factors)},
            "robust_references": {
                "norm": norm_reference,
                "anchor_harm": harm_reference,
                "perturbation_sensitivity": sensitivity_reference,
            },
        },
    )

