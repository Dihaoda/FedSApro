from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


ATTACKS = ("none", "label_flip", "textual_backdoor")
METHODS = ("fedavg", "indexguard", "flame", "fltrust", "safefedllm", "fedgraph")
FIELDS = (
    "model",
    "attack",
    "method",
    "seed",
    "standard_accuracy",
    "heterogeneous_accuracy",
    "mean_accuracy",
    "domain_gap",
    "backdoor_asr_standard",
    "backdoor_asr_heterogeneous",
    "mean_backdoor_asr",
    "mean_accuracy_delta_vs_fedavg",
    "heterogeneous_accuracy_delta_vs_fedavg",
    "domain_gap_delta_vs_fedavg",
    "mean_backdoor_asr_delta_vs_fedavg",
    "attacker_rejection_rate",
    "heterogeneous_retention_rate",
    "run_dir",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the aggressive six-method study")
    parser.add_argument("--root", default="results/aggressive_security_study_v1")
    parser.add_argument("--output", default="reports/aggressive_security_study_v1")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_optional(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return (float(left) + float(right)) / 2.0


def _selection_rates(path: Path) -> tuple[float | None, float | None]:
    rows = _read_csv(path / "client_rounds.csv")
    attackers = [int(row["selected"]) for row in rows if row["role"] == "attacker"]
    heterogeneous = [int(row["selected"]) for row in rows if row["role"] == "heterogeneous"]
    attacker_rejection = (
        100.0 * sum(value == 0 for value in attackers) / len(attackers) if attackers else None
    )
    heterogeneous_retention = (
        100.0 * sum(value == 1 for value in heterogeneous) / len(heterogeneous)
        if heterogeneous
        else None
    )
    return attacker_rejection, heterogeneous_retention


def build_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in root.rglob("summary.json"):
        summary = _read_json(path)
        if summary.get("method") not in METHODS:
            continue
        measured = summary["measured"]
        standard = float(measured["standard_accuracy"])
        heterogeneous = float(measured["heterogeneous_accuracy"])
        asr_standard = measured.get("backdoor_asr_standard")
        asr_heterogeneous = measured.get("backdoor_asr_heterogeneous")
        attacker_rejection, heterogeneous_retention = _selection_rates(path.parent)
        rows.append(
            {
                "model": summary["model"],
                "attack": summary["attack"],
                "method": summary["method"],
                "seed": int(summary["seed"]),
                "standard_accuracy": standard,
                "heterogeneous_accuracy": heterogeneous,
                "mean_accuracy": (standard + heterogeneous) / 2.0,
                "domain_gap": float(measured["domain_gap"]),
                "backdoor_asr_standard": asr_standard,
                "backdoor_asr_heterogeneous": asr_heterogeneous,
                "mean_backdoor_asr": _mean_optional(asr_standard, asr_heterogeneous),
                "attacker_rejection_rate": attacker_rejection,
                "heterogeneous_retention_rate": heterogeneous_retention,
                "run_dir": str(path.parent.relative_to(root)),
            }
        )
    by_cell = {
        (row["model"], row["attack"], row["method"], row["seed"]): row for row in rows
    }
    for row in rows:
        fedavg = by_cell.get((row["model"], row["attack"], "fedavg", row["seed"]))
        row["mean_accuracy_delta_vs_fedavg"] = (
            row["mean_accuracy"] - fedavg["mean_accuracy"] if fedavg else None
        )
        row["heterogeneous_accuracy_delta_vs_fedavg"] = (
            row["heterogeneous_accuracy"] - fedavg["heterogeneous_accuracy"] if fedavg else None
        )
        row["domain_gap_delta_vs_fedavg"] = (
            row["domain_gap"] - fedavg["domain_gap"] if fedavg else None
        )
        row["mean_backdoor_asr_delta_vs_fedavg"] = (
            row["mean_backdoor_asr"] - fedavg["mean_backdoor_asr"]
            if fedavg and row["mean_backdoor_asr"] is not None
            else None
        )
    return sorted(
        rows,
        key=lambda row: (
            ATTACKS.index(row["attack"]),
            METHODS.index(row["method"]),
            row["seed"],
        ),
    )


def conclusion_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell = {(row["attack"], row["method"]): row for row in rows}
    observed = set(by_cell)
    expected = {(attack, method) for attack in ATTACKS for method in METHODS}
    clean = by_cell.get(("none", "fedavg"))
    checks: dict[str, Any] = {
        "matrix_complete": observed == expected,
        "observed_cell_count": len(observed),
        "expected_cell_count": len(expected),
        "missing_cells": [list(cell) for cell in sorted(expected - observed)],
        "thresholds": {
            "clean_standard_accuracy_min": 85.0,
            "clean_domain_gap_min": 15.0,
            "attacked_standard_accuracy_min": 75.0,
            "attacked_gap_min": {"label_flip": 30.0, "textual_backdoor": 25.0},
            "attacked_gap_increase_min": 10.0,
            "defense_mean_accuracy_gain_min": 5.0,
            "defense_heterogeneous_accuracy_gain_min": 10.0,
            "defense_gap_reduction_min": 8.0,
            "textual_asr_reduction_min": 30.0,
            "clean_defense_mean_loss_max": 2.0,
        },
    }
    checks["clean_baseline"] = (
        {
            "standard_accuracy": clean["standard_accuracy"],
            "domain_gap": clean["domain_gap"],
            "passes": clean["standard_accuracy"] >= 85.0 and clean["domain_gap"] >= 15.0,
        }
        if clean
        else {"passes": False}
    )
    attack_checks = []
    defense_checks = []
    clean_defense_checks = []
    for method in METHODS[1:]:
        row = by_cell.get(("none", method))
        if row and clean:
            loss = clean["mean_accuracy"] - row["mean_accuracy"]
            clean_defense_checks.append(
                {"method": method, "mean_accuracy_loss": loss, "passes": loss <= 2.0}
            )
    for attack in ATTACKS[1:]:
        baseline = by_cell.get((attack, "fedavg"))
        if baseline and clean:
            increase = baseline["domain_gap"] - clean["domain_gap"]
            attack_checks.append(
                {
                    "attack": attack,
                    "standard_accuracy": baseline["standard_accuracy"],
                    "domain_gap": baseline["domain_gap"],
                    "gap_increase": increase,
                    "passes": baseline["standard_accuracy"] >= 75.0
                    and baseline["domain_gap"]
                    >= checks["thresholds"]["attacked_gap_min"][attack]
                    and increase >= 10.0,
                }
            )
        for method in METHODS[1:]:
            row = by_cell.get((attack, method))
            if not row or not baseline:
                continue
            asr_reduction = (
                baseline["mean_backdoor_asr"] - row["mean_backdoor_asr"]
                if attack == "textual_backdoor"
                else None
            )
            passes = (
                row["mean_accuracy"] - baseline["mean_accuracy"] >= 5.0
                and row["heterogeneous_accuracy"] - baseline["heterogeneous_accuracy"] >= 10.0
                and baseline["domain_gap"] - row["domain_gap"] >= 8.0
                and (asr_reduction is None or asr_reduction >= 30.0)
            )
            defense_checks.append(
                {
                    "attack": attack,
                    "method": method,
                    "mean_accuracy_gain": row["mean_accuracy"] - baseline["mean_accuracy"],
                    "heterogeneous_accuracy_gain": row["heterogeneous_accuracy"]
                    - baseline["heterogeneous_accuracy"],
                    "domain_gap_reduction": baseline["domain_gap"] - row["domain_gap"],
                    "mean_asr_reduction": asr_reduction,
                    "passes": passes,
                }
            )
    checks["clean_defense_checks"] = clean_defense_checks
    checks["attack_baseline_checks"] = attack_checks
    checks["defense_checks"] = defense_checks
    checks["all_predeclared_checks_pass"] = (
        checks["matrix_complete"]
        and checks["clean_baseline"]["passes"]
        and len(clean_defense_checks) == 5
        and all(item["passes"] for item in clean_defense_checks)
        and len(attack_checks) == 2
        and all(item["passes"] for item in attack_checks)
        and len(defense_checks) == 10
        and all(item["passes"] for item in defense_checks)
    )
    return checks


def _fmt(value: Any, signed: bool = False) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):+.2f}" if signed else f"{float(value):.2f}"


def _write_report(path: Path, rows: list[dict[str, Any]], checks: dict[str, Any]) -> None:
    lines = [
        "# Aggressive Qwen2.5-0.5B security study",
        "",
        "All values are measured single-seed results from this study. No paper target values, HSSR, or HRCW are used.",
        "",
        "| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Gap | Mean Δ vs FedAvg | Heterogeneous Δ | Gap Δ | ASR standard | ASR heterogeneous | ASR mean | ASR Δ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['attack']} | {row['method']} | {_fmt(row['standard_accuracy'])} | "
            f"{_fmt(row['heterogeneous_accuracy'])} | {_fmt(row['mean_accuracy'])} | "
            f"{_fmt(row['domain_gap'])} | {_fmt(row['mean_accuracy_delta_vs_fedavg'], True)} | "
            f"{_fmt(row['heterogeneous_accuracy_delta_vs_fedavg'], True)} | "
            f"{_fmt(row['domain_gap_delta_vs_fedavg'], True)} | "
            f"{_fmt(row['backdoor_asr_standard'])} | {_fmt(row['backdoor_asr_heterogeneous'])} | "
            f"{_fmt(row['mean_backdoor_asr'])} | "
            f"{_fmt(row['mean_backdoor_asr_delta_vs_fedavg'], True)} |"
        )
    passed = sum(item["passes"] for item in checks.get("defense_checks", []))
    lines.extend(
        [
            "",
            "## Predeclared aggressive checks",
            "",
            f"- Complete matrix: {checks['observed_cell_count']}/{checks['expected_cell_count']} cells.",
            f"- Attacked defense cells passing every utility/gap/ASR bar: {passed}/{len(checks.get('defense_checks', []))}.",
            f"- All predeclared checks pass: {checks['all_predeclared_checks_pass']}.",
            "",
            "Exact thresholds and per-cell decisions are in `conclusion_checks.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _copy_evidence(root: Path, output: Path, rows: list[dict[str, Any]]) -> None:
    names = (
        "summary.json",
        "resolved_config.yaml",
        "provenance.json",
        "client_rounds.csv",
        "round_metrics.csv",
        "aggregation_diagnostics.jsonl",
    )
    for row in rows:
        source = root / str(row["run_dir"])
        destination = output / "evidence" / str(row["run_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            source_file = source / name
            if source_file.exists():
                shutil.copy2(source_file, destination / name)


def main() -> None:
    args = _arguments()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = build_rows(root)
    checks = conclusion_checks(rows)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in rows)
    (output / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "conclusion_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output / "REPORT.md", rows, checks)
    _copy_evidence(root, output, rows)
    print(f"Exported {len(rows)} runs; aggressive checks pass={checks['all_predeclared_checks_pass']}")


if __name__ == "__main__":
    main()
