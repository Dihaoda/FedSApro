from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .data import STANDARD
from .state import ParameterLayout, gather_first_lora_b, gather_qv


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


def scaffold(updates: Sequence[ClientUpdate], *, server_lr: float) -> AggregationResult:
    result = fedavg(updates, norm_aware=True)
    result.delta.mul_(float(server_lr))
    result.diagnostics["server_lr"] = float(server_lr)
    return result


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
    norm_ratio_cap = float(config.get("norm_ratio_cap", 0.0))
    if norm_ratio_cap > 0:
        norms = [float(torch.linalg.vector_norm(update.delta.float()).item()) for update in updates]
        median_norm = max(float(np.median(norms)), 1e-12)
        norm_retained = [norm <= median_norm * norm_ratio_cap for norm in norms]
        retained = [
            cluster_keep and norm_keep
            for cluster_keep, norm_keep in zip(retained, norm_retained)
        ]
        diagnostics["norm_ratio_cap"] = norm_ratio_cap
        diagnostics["median_update_norm"] = median_norm
        diagnostics["norm_rejected_clients"] = [
            update.client_id for update, keep in zip(updates, norm_retained) if not keep
        ]
    raw = _sample_coefficients(updates)
    rejected_weight = float(config.get("rejected_weight", 0.0))
    effective = {
        update.client_id: raw[update.client_id] * (1.0 if keep else rejected_weight)
        for update, keep in zip(updates, retained)
    }
    if not any(retained):
        retained = [True] * len(updates)
        effective = dict(raw)
        diagnostics["all_rejected_fallback"] = True
    diagnostics["ios_matrix"] = similarity.tolist()
    diagnostics["rejected_weight"] = rejected_weight
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
    rejected_weight = float(config.get("rejected_weight", 0.0))
    aggregation_coefficients = {
        update.client_id: raw[update.client_id] * (1.0 if keep else rejected_weight)
        for update, keep in zip(updates, retained)
    }
    effective = {
        update.client_id: raw[update.client_id]
        * clip_scales[update.client_id]
        * (1.0 if keep else rejected_weight)
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
            "rejected_weight": rejected_weight,
        },
    )


def fltrust(
    updates: Sequence[ClientUpdate],
    root_delta: torch.Tensor,
    config: Mapping[str, Any] | None = None,
) -> AggregationResult:
    root = root_delta.float().cpu()
    root_norm = max(float(torch.linalg.vector_norm(root).item()), 1e-12)
    raw = _sample_coefficients(updates)
    trust: dict[int, float] = {}
    norm_scales: dict[int, float] = {}
    normalised: list[ClientUpdate] = []
    trust_floor = float((config or {}).get("trust_floor", 0.0))
    for update in updates:
        value = update.delta.float().cpu()
        norm = max(float(torch.linalg.vector_norm(value).item()), 1e-12)
        cosine = float(torch.dot(value, root).item() / (norm * root_norm))
        score = max(trust_floor, cosine)
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
            "trust_floor": trust_floor,
        },
    )


def _load_safefedllm_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    probe_path = str(config.get("probe_path", "")).strip()
    if not probe_path:
        raise ValueError("Safe-FedLLM requires method.safefedllm.probe_path")
    try:
        probe = torch.load(probe_path, map_location="cpu", weights_only=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Safe-FedLLM probe not found: {probe_path}") from exc
    if not isinstance(probe, dict) or probe.get("schema_version") != 1:
        raise ValueError("Unsupported Safe-FedLLM probe artifact")
    return probe


def safefedllm(
    updates: Sequence[ClientUpdate], layout: ParameterLayout, config: Mapping[str, Any]
) -> AggregationResult:
    """Safe-FedLLM Shadow-Level weighting with a held-out terminal-delta probe.

    The official method classifies intermediate LoRA deltas and maps the harmful
    step ratio ``r`` to ``(1-r)^eta``. This federated classifier harness exposes
    one terminal delta per client, so a held-out linear probe's harmful
    probability is used as the soft single-snapshot ratio. Probe fitting is kept
    outside the formal seed and the artifact records its calibration provenance.
    """
    probe = _load_safefedllm_probe(config)
    weight = torch.as_tensor(probe["coefficient"], dtype=torch.float32).reshape(-1)
    bias = float(probe["intercept"])
    feature_dim = int(probe["feature_dim"])
    eta = float(config.get("eta", 7.0))
    minimum = float(config.get("min_weight", 0.01))
    threshold = float(config.get("selection_threshold", 0.5))
    raw = _sample_coefficients(updates)
    probabilities: dict[int, float] = {}
    security: dict[int, float] = {}
    effective: dict[int, float] = {}
    selected: dict[int, bool] = {}
    magnitudes: dict[int, float] = {}
    for update in updates:
        feature = gather_first_lora_b(update.delta.float().cpu(), layout)
        magnitude = torch.linalg.vector_norm(feature)
        magnitudes[update.client_id] = float(magnitude.item())
        feature = feature / magnitude.clamp_min(1e-12)
        feature_policy = str(probe.get("feature_policy", "first_transformer_layer_lora_B_l2"))
        if feature_policy == "first_transformer_layer_lora_B_l2_plus_log_norm":
            feature = torch.cat((feature, torch.log1p(magnitude).reshape(1)))
        elif feature_policy != "first_transformer_layer_lora_B_l2":
            raise ValueError(f"Unsupported Safe-FedLLM feature policy: {feature_policy}")
        if feature.numel() != feature_dim or weight.numel() != feature_dim:
            raise ValueError(
                f"Safe-FedLLM probe dimension {feature_dim} does not match update "
                f"feature dimension {feature.numel()}"
            )
        probability = float(torch.sigmoid(torch.dot(feature, weight) + bias).item())
        factor = max(minimum, (1.0 - probability) ** eta)
        probabilities[update.client_id] = probability
        security[update.client_id] = factor
        effective[update.client_id] = raw[update.client_id] * factor
        selected[update.client_id] = probability < threshold
    norm_guard_fraction = float(config.get("norm_guard_fraction", 0.0))
    norm_guard_count = min(
        len(updates), int(math.ceil(len(updates) * norm_guard_fraction))
    )
    norm_guard_clients = sorted(
        magnitudes, key=lambda client_id: (-magnitudes[client_id], client_id)
    )[:norm_guard_count]
    for client_id in norm_guard_clients:
        security[client_id] = minimum
        effective[client_id] = raw[client_id] * minimum
        selected[client_id] = False
    return AggregationResult(
        delta=_weighted_mean(updates, effective),
        raw_coefficients=raw,
        effective_coefficients=effective,
        selected=selected,
        diagnostics={
            "variant": "shadow-level-terminal-delta",
            "probe_path": str(config.get("probe_path")),
            "probe_calibration": probe.get("calibration"),
            "harmful_probability": probabilities,
            "security_factor": security,
            "norm_guard_fraction": norm_guard_fraction,
            "norm_guard_clients": norm_guard_clients,
            "eta": eta,
        },
    )


def fedgraph(
    updates: Sequence[ClientUpdate],
    config: Mapping[str, Any],
    layout: ParameterLayout | None = None,
) -> AggregationResult:
    """Graph-centrality clustering defense from FedGraph.

    Clients are connected when their cosine similarity exceeds the configured
    threshold. Degree, betweenness, and closeness centralities form the paper's
    fingerprint, and HDBSCAN marks points outside the benign-majority cluster as
    outliers. An all-outlier safety guard keeps every client when no valid cluster
    can be formed.
    """
    import networkx as nx
    from sklearn.cluster import HDBSCAN

    size = len(updates)
    if size < 3:
        return fedavg(updates)
    feature_scope = str(config.get("feature_scope", "full_delta"))
    if feature_scope == "first_lora_b":
        if layout is None:
            raise ValueError("FedGraph first_lora_b feature scope requires a parameter layout")
        matrix = torch.stack(
            [gather_first_lora_b(update.delta.float().cpu(), layout) for update in updates]
        )
    elif feature_scope == "full_delta":
        matrix = torch.stack([update.delta.float().cpu() for update in updates])
    else:
        raise ValueError(f"Unsupported FedGraph feature scope: {feature_scope}")
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    similarity = (matrix @ matrix.T).clamp(-1.0, 1.0).numpy()
    threshold = float(config.get("similarity_threshold", 0.8))
    min_neighbors = max(0, int(config.get("min_neighbors", 1)))
    graph = nx.Graph()
    graph.add_nodes_from(range(size))
    for left in range(size):
        for right in range(left + 1, size):
            if similarity[left, right] >= threshold:
                graph.add_edge(left, right, weight=float(similarity[left, right]))
    for left in range(size):
        neighbors = set(graph.neighbors(left))
        if len(neighbors) >= min_neighbors:
            continue
        candidates = np.argsort(-similarity[left])
        for right in candidates:
            right = int(right)
            if right == left or right in neighbors:
                continue
            graph.add_edge(left, right, weight=float(similarity[left, right]))
            neighbors.add(right)
            if len(neighbors) >= min_neighbors:
                break

    degree = dict(graph.degree())
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)
    closeness = nx.closeness_centrality(graph)
    features = np.asarray(
        [[degree[index], betweenness[index], closeness[index]] for index in range(size)],
        dtype=np.float64,
    )
    min_cluster_size = int(config.get("min_cluster_size", size // 2 + 1))
    min_cluster_size = min(size, max(2, min_cluster_size))
    min_samples = max(1, int(config.get("min_samples", 1)))
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        allow_single_cluster=bool(config.get("allow_single_cluster", True)),
        copy=True,
    ).fit_predict(features)
    counts = {int(label): int(np.sum(labels == label)) for label in np.unique(labels)}
    retained = [int(label) != -1 for label in labels]
    fallback = not any(retained)
    if fallback:
        retained = [True] * size
    raw = _sample_coefficients(updates)
    rejected_weight = float(config.get("rejected_weight", 0.0))
    effective = {
        update.client_id: raw[update.client_id] * (1.0 if keep else rejected_weight)
        for update, keep in zip(updates, retained)
    }
    return AggregationResult(
        delta=_weighted_mean(updates, effective),
        raw_coefficients=raw,
        effective_coefficients=effective,
        selected={update.client_id: keep for update, keep in zip(updates, retained)},
        diagnostics={
            "feature_scope": feature_scope,
            "similarity_threshold": threshold,
            "similarity_matrix": similarity.tolist(),
            "centrality_features": features.tolist(),
            "cluster_labels": labels.tolist(),
            "cluster_sizes": counts,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "outlier_count": int(np.sum(labels == -1)),
            "all_outlier_fallback": fallback,
            "rejected_weight": rejected_weight,
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
    if method == "fedprox":
        return fedavg(updates, norm_aware=True)
    if method == "scaffold":
        return scaffold(updates, server_lr=float(method_config["scaffold_server_lr"]))
    if method == "indexguard":
        return indexguard(updates, layout, method_config["indexguard"])
    if method == "flame":
        if generator is None:
            generator = torch.Generator().manual_seed(0)
        return flame(updates, method_config["flame"], generator=generator)
    if method == "fltrust":
        if root_delta is None:
            raise ValueError("FLTrust requires a server root update")
        return fltrust(updates, root_delta, method_config.get("fltrust", {}))
    if method == "safefedllm":
        return safefedllm(updates, layout, method_config["safefedllm"])
    if method == "fedgraph":
        return fedgraph(updates, method_config["fedgraph"], layout)
    raise ValueError(f"Unknown aggregation method: {method}")


