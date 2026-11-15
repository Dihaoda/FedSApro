from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .data import STANDARD
from .state import ParameterLayout, gather_qv


@dataclass
class ClientUpdate:
    client_id: int
    role: str
    delta: torch.Tensor
    sample_count: int


@dataclass
class AggregationResult:
    delta: torch.Tensor
    raw_coefficients: dict[int, float]
    effective_coefficients: dict[int, float]
    selected: dict[int, bool]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _sample_coefficients(updates: Sequence[ClientUpdate]) -> dict[int, float]:
    return {update.client_id: float(update.sample_count) for update in updates}


def _weighted_mean(
    updates: Sequence[ClientUpdate], coefficients: Mapping[int, float]
) -> torch.Tensor:
    if not updates:
        raise ValueError("Cannot aggregate an empty update list")
    total = sum(max(0.0, float(coefficients.get(update.client_id, 0.0))) for update in updates)
    if total <= 0:
        raise RuntimeError("Every client contribution coefficient is zero")
    result = torch.zeros_like(updates[0].delta, dtype=torch.float32, device="cpu")
    for update in updates:
        coefficient = max(0.0, float(coefficients.get(update.client_id, 0.0)))
        result.add_(update.delta.float().cpu(), alpha=coefficient / total)
    return result


def _norm_aware_coefficients(updates: Sequence[ClientUpdate]) -> dict[int, float]:
    standard_norms = [
        float(torch.linalg.vector_norm(update.delta.float()).item())
        for update in updates
        if update.role == STANDARD
    ]
    all_norms = [float(torch.linalg.vector_norm(update.delta.float()).item()) for update in updates]
    reference = float(np.median(standard_norms or all_norms))
    reference = max(reference, 1e-12)
    return {
        update.client_id: float(update.sample_count)
        * float(torch.linalg.vector_norm(update.delta.float()).item())
        / reference
        for update in updates
    }


def fedavg(updates: Sequence[ClientUpdate], *, norm_aware: bool = False) -> AggregationResult:
    raw = _sample_coefficients(updates)
    effective = _norm_aware_coefficients(updates) if norm_aware else dict(raw)
    return AggregationResult(
        delta=_weighted_mean(updates, raw),
        raw_coefficients=raw,
        effective_coefficients=effective,
        selected={update.client_id: True for update in updates},
    )


def topk_indices(vector: torch.Tensor, fraction: float) -> list[int]:
    if not 0 < fraction <= 1:
        raise ValueError("topk fraction must be in (0, 1]")
    count = max(1, min(vector.numel(), int(round(vector.numel() * fraction))))
    return torch.topk(vector.abs(), k=count, largest=True, sorted=True).indices.cpu().tolist()


def index_overlap_similarity(index_lists: Sequence[Sequence[int]]) -> torch.Tensor:
    size = len(index_lists)
    if size == 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    budget = max(1, len(index_lists[0]))
    sets = [set(values) for values in index_lists]
    similarity = torch.eye(size, dtype=torch.float32)
    for left in range(size):
        for right in range(left + 1, size):
            value = len(sets[left].intersection(sets[right])) / float(budget)
            similarity[left, right] = value
            similarity[right, left] = value
    return similarity


def _agglomerative_labels(distance: np.ndarray, clusters: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    kwargs = {"n_clusters": max(2, min(int(clusters), len(distance))), "linkage": "average"}
    try:
        estimator = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:  # scikit-learn < 1.2 compatibility
        estimator = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return estimator.fit_predict(distance)


def _cohesive_minority_mask(
    similarity: torch.Tensor,
    *,
    clusters: int,
    cohesion_margin: float,
    minority_ratio: float,
    fallback: str,
) -> tuple[list[bool], dict[str, Any]]:
    labels = _agglomerative_labels((1.0 - similarity).clamp_min(0).numpy(), clusters)
    groups = {int(label): np.flatnonzero(labels == label).tolist() for label in np.unique(labels)}
    majority = sorted(groups, key=lambda label: (-len(groups[label]), label))[0]
    majority_size = len(groups[majority])

    def cohesion(members: list[int]) -> float:
        if len(members) < 2:
            return float("nan")
        block = similarity[np.ix_(members, members)]
        return float((block.sum() - block.diag().sum()).item() / (len(members) * (len(members) - 1)))

    cohesions = {label: cohesion(members) for label, members in groups.items()}
    finite = [value for value in cohesions.values() if math.isfinite(value)]
    median = float(np.median(finite)) if finite else float("nan")
    suspicious: list[int] = []
    for label, members in groups.items():
        if label == majority or len(members) < 2 or not math.isfinite(median):
            continue
        if cohesions[label] >= median + cohesion_margin and len(members) <= minority_ratio * majority_size:
            suspicious.append(label)
    if not suspicious and fallback != "none":
        candidates = [label for label in groups if label != majority]
        if candidates:
            if fallback == "max_cohesion":
                suspicious = [
                    max(candidates, key=lambda label: cohesions[label] if math.isfinite(cohesions[label]) else -1)
                ]
            elif fallback == "second_largest":
                suspicious = [sorted(candidates, key=lambda label: (-len(groups[label]), label))[0]]
            else:
                raise ValueError(f"Unknown IndexGuard fallback: {fallback}")
    mask = [int(label) not in suspicious for label in labels]
    diagnostics = {
        "cluster_labels": labels.tolist(),
        "cluster_members": groups,
        "cluster_cohesion": cohesions,
        "majority_cluster": majority,
        "suspicious_clusters": suspicious,
    }
    return mask, diagnostics


def indexguard(
    updates: Sequence[ClientUpdate], layout: ParameterLayout, config: Mapping[str, Any]
) -> AggregationResult:
    qv_updates = [gather_qv(update.delta, layout) for update in updates]
    signatures = [topk_indices(value, float(config["topk_fraction"])) for value in qv_updates]
    similarity = index_overlap_similarity(signatures)
    retained, diagnostics = _cohesive_minority_mask(
        similarity,
        clusters=int(config["clusters"]),
        cohesion_margin=float(config["cohesion_margin"]),
        minority_ratio=float(config["minority_ratio"]),
        fallback=str(config.get("fallback", "none")),
    )
    raw = _sample_coefficients(updates)
    effective = {
        update.client_id: raw[update.client_id] if keep else 0.0
        for update, keep in zip(updates, retained)
    }
    if not any(retained):
        retained = [True] * len(updates)
        effective = dict(raw)
        diagnostics["all_rejected_fallback"] = True
    diagnostics["ios_matrix"] = similarity.tolist()
    return AggregationResult(
        delta=_weighted_mean(updates, effective),
        raw_coefficients=raw,
        effective_coefficients=effective,
        selected={update.client_id: keep for update, keep in zip(updates, retained)},
        diagnostics=diagnostics,
    )


def flame(
    updates: Sequence[ClientUpdate], config: Mapping[str, Any], *, generator: torch.Generator
) -> AggregationResult:
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics.pairwise import cosine_distances

    matrix = torch.stack([update.delta.float().cpu() for update in updates]).numpy()
    distance = cosine_distances(matrix)
    estimator = HDBSCAN(
        metric="precomputed",
        min_cluster_size=max(2, int(config["min_cluster_size"])),
        min_samples=max(1, int(config["min_samples"])),
        allow_single_cluster=True,
    )
    labels = estimator.fit_predict(distance)
    non_noise = [int(label) for label in np.unique(labels) if int(label) >= 0]
    if non_noise:
        retained_label = sorted(non_noise, key=lambda label: (-int((labels == label).sum()), label))[0]
        retained = [int(label) == retained_label for label in labels]
    else:
        retained_label = None
        retained = [True] * len(updates)

    norms = torch.tensor([torch.linalg.vector_norm(update.delta.float()) for update in updates])
    clip_norm = float(torch.median(norms).item())
    clipped: list[ClientUpdate] = []
    clip_scales: dict[int, float] = {}
    for update in updates:
        norm = max(float(torch.linalg.vector_norm(update.delta.float()).item()), 1e-12)
        scale = min(1.0, clip_norm / norm)
        clip_scales[update.client_id] = scale
        clipped.append(
            ClientUpdate(update.client_id, update.role, update.delta.float() * scale, update.sample_count)
        )
    raw = _sample_coefficients(updates)
    aggregation_coefficients = {
        update.client_id: raw[update.client_id] if keep else 0.0
        for update, keep in zip(updates, retained)
    }
    effective = {
        update.client_id: raw[update.client_id] * clip_scales[update.client_id] if keep else 0.0
        for update, keep in zip(updates, retained)
    }
    # FLAME applies clipping before the retained clients are averaged.  Use the
    # unscaled sample coefficients for the actual aggregate; ``effective`` is
    # the equivalent coefficient on each original, unclipped update and is
    # therefore the value required by HRCW/HSSR diagnostics.
    aggregate = _weighted_mean(clipped, aggregation_coefficients)
    multiplier = float(config.get("noise_multiplier", 0.0))
    if multiplier > 0:
        coordinate_std = multiplier * clip_norm / math.sqrt(max(1, aggregate.numel()))
        noise = torch.randn(aggregate.shape, generator=generator, dtype=aggregate.dtype)
        aggregate.add_(noise, alpha=coordinate_std)
    return AggregationResult(
        delta=aggregate,
        raw_coefficients=raw,
        effective_coefficients=effective,
        selected={update.client_id: keep for update, keep in zip(updates, retained)},
        diagnostics={
            "cluster_labels": labels.tolist(),
            "retained_cluster": retained_label,
            "clip_norm": clip_norm,
            "clip_scales": clip_scales,
        },
    )


def fltrust(updates: Sequence[ClientUpdate], root_delta: torch.Tensor) -> AggregationResult:
    root = root_delta.float().cpu()
    root_norm = max(float(torch.linalg.vector_norm(root).item()), 1e-12)
    raw = _sample_coefficients(updates)
    trust: dict[int, float] = {}
    norm_scales: dict[int, float] = {}
    normalised: list[ClientUpdate] = []
    for update in updates:
        value = update.delta.float().cpu()
        norm = max(float(torch.linalg.vector_norm(value).item()), 1e-12)
        cosine = float(torch.dot(value, root).item() / (norm * root_norm))
        score = max(0.0, cosine)
        trust[update.client_id] = score
        norm_scales[update.client_id] = root_norm / norm
        normalised.append(
            ClientUpdate(
                update.client_id,
                update.role,
                value * (root_norm / norm),
                update.sample_count,
            )
        )
    if sum(trust.values()) <= 0:
        trust = dict(raw)
        fallback = True
    else:
        fallback = False
    effective = {
        update.client_id: trust[update.client_id] * norm_scales[update.client_id]
        for update in updates
    }
    return AggregationResult(
        delta=_weighted_mean(normalised, trust),
        raw_coefficients=raw,
        effective_coefficients=effective,
        selected={update.client_id: trust[update.client_id] > 0 for update in updates},
        diagnostics={
            "root_norm": root_norm,
            "norm_scales": norm_scales,
            "zero_trust_fallback": fallback,
        },
    )


def aggregate(
    method: str,
    updates: Sequence[ClientUpdate],
    *,
    layout: ParameterLayout,
    method_config: Mapping[str, Any],
    root_delta: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> AggregationResult:
    if method == "fedavg":
        return fedavg(updates)
    if method in {"fedprox", "scaffold"}:
        return fedavg(updates, norm_aware=True)
    if method == "indexguard":
        return indexguard(updates, layout, method_config["indexguard"])
    if method == "flame":
        if generator is None:
            generator = torch.Generator().manual_seed(0)
        return flame(updates, method_config["flame"], generator=generator)
    if method == "fltrust":
        if root_delta is None:
            raise ValueError("FLTrust requires a server root update")
        return fltrust(updates, root_delta)
    raise ValueError(f"Unknown aggregation method: {method}")

