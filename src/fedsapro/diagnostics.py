from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .config import get_path, load_yaml
from .data import HETEROGENEOUS, STANDARD
from .utils import atomic_json_dump


def contribution_metrics(
    rows: Sequence[Mapping[str, Any]], *, tau: float = 0.25, epsilon: float = 1e-12
) -> dict[str, float]:
    by_round: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_round[int(row["round"])].append(row)

    hrcw_values: list[float] = []
    suppressed = 0
    heterogeneous_total = 0
    for round_rows in by_round.values():
        standard = [
            float(row["effective_coefficient"])
            for row in round_rows
            if row["role"] == STANDARD
        ]
        heterogeneous = [
            float(row["effective_coefficient"])
            for row in round_rows
            if row["role"] == HETEROGENEOUS
        ]
        if not standard or not heterogeneous:
            continue
        reference = median(standard)
        hrcw_values.append(median(heterogeneous) / (reference + epsilon))
        threshold = tau * reference
        suppressed += sum(value < threshold for value in heterogeneous)
        heterogeneous_total += len(heterogeneous)
    return {
        "hrcw": sum(hrcw_values) / max(1, len(hrcw_values)),
        "hssr": suppressed / max(1, heterogeneous_total),
        "rounds_observed": float(len(hrcw_values)),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def domain_gap_points(standard_accuracy: float, heterogeneous_accuracy: float) -> float:
    """Return the absolute cross-domain accuracy disparity in percentage points."""
    return abs(standard_accuracy - heterogeneous_accuracy) * 100.0


def summarize_run(config: Mapping[str, Any], run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir)
    client_rows = _read_csv(directory / "client_rounds.csv")
    round_rows = _read_csv(directory / "round_metrics.csv")
    if not round_rows:
        raise RuntimeError(f"No measured round metrics in {directory}")
    measured_rows = [row for row in round_rows if row.get("standard_accuracy")]
    if not measured_rows:
        raise RuntimeError(f"No evaluated rounds in {directory}")
    final = measured_rows[-1]
    contribution = contribution_metrics(
        client_rows,
        tau=float(get_path(config, "diagnostics.hssr_tau")),
        epsilon=float(get_path(config, "diagnostics.coefficient_epsilon")),
    )
    standard_accuracy = float(final["standard_accuracy"])
    heterogeneous_accuracy = float(final["heterogeneous_accuracy"])
    model = str(get_path(config, "model.key"))
    attack = str(get_path(config, "attack.name"))
    method = str(get_path(config, "method.name"))
    target_path = Path(str(get_path(config, "reporting.target_file")))
    targets = load_yaml(target_path)
    aliases = config["reporting"].get("target_model_aliases", {})
    target_model = str(aliases.get(model, model))
    target = targets.get(target_model, {}).get(attack, {}).get(method)
    measured = {
        "hssr": contribution["hssr"] * 100.0,
        "hrcw": contribution["hrcw"],
        "standard_accuracy": standard_accuracy * 100.0,
        "heterogeneous_accuracy": heterogeneous_accuracy * 100.0,
        # An attack can reverse which domain is better without eliminating the
        # cross-domain disparity.
        "domain_gap": domain_gap_points(standard_accuracy, heterogeneous_accuracy),
        "backdoor_asr_standard": (
            float(final["backdoor_asr_standard"]) * 100.0
            if final.get("backdoor_asr_standard")
            else None
        ),
        "backdoor_asr_heterogeneous": (
            float(final["backdoor_asr_heterogeneous"]) * 100.0
            if final.get("backdoor_asr_heterogeneous")
            else None
        ),
    }
    differences = (
        {
            key: measured[key] - float(target[key])
            for key in ("hssr", "standard_accuracy", "heterogeneous_accuracy")
        }
        if target is not None
        else None
    )
    accuracy_tolerance = float(get_path(config, "reporting.accuracy_tolerance_points"))
    hssr_tolerance = float(get_path(config, "reporting.hssr_tolerance_points"))
    accepted = (
        abs(differences["standard_accuracy"]) <= accuracy_tolerance
        and abs(differences["heterogeneous_accuracy"]) <= accuracy_tolerance
        and abs(differences["hssr"]) <= hssr_tolerance
        if differences is not None
        else None
    )
    summary = {
        "run_class": str(get_path(config, "reporting.run_class")),
        "model": model,
        "target_reference_model": target_model,
        "attack": attack,
        "method": method,
        "seed": int(get_path(config, "experiment.seed")),
        "final_round": int(final["round"]),
        "measured": measured,
        "target": target,
        "difference_points": differences,
        "tolerance_points": {"accuracy": accuracy_tolerance, "hssr": hssr_tolerance},
        "accepted": accepted,
    }
    atomic_json_dump(summary, directory / "summary.json")
    return summary
