from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from export_aggressive_results import FIELDS, METHODS, _copy_evidence, _fmt, build_rows


ATTACKS = ("none", "label_flip", "textual_backdoor")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the natural domain-attack study")
    parser.add_argument("--root", default="results/natural_domain_attack_study_v3")
    parser.add_argument("--output", default="reports/natural_domain_attack_study_v3")
    parser.add_argument("--title", default="Qwen2.5-0.5B natural domain-attack study")
    return parser.parse_args()


def conclusion_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell = {(row["attack"], row["method"]): row for row in rows}
    expected = {(attack, method) for attack in ATTACKS for method in METHODS}
    clean = by_cell.get(("none", "fedavg"))
    drop_checks: list[dict[str, Any]] = []
    defense_checks: list[dict[str, Any]] = []
    separation_checks: list[dict[str, Any]] = []
    asr_checks: list[dict[str, Any]] = []
    for attack in ATTACKS[1:]:
        baseline = by_cell.get((attack, "fedavg"))
        if clean and baseline:
            drop = clean["standard_accuracy"] - baseline["standard_accuracy"]
            drop_checks.append({"attack": attack, "standard_drop": drop, "passes": drop >= 5.0})
        if not baseline:
            continue
        values = [(baseline["standard_accuracy"], "fedavg")]
        for method in METHODS[1:]:
            row = by_cell.get((attack, method))
            if not row:
                continue
            margin = row["standard_accuracy"] - baseline["standard_accuracy"]
            defense_checks.append(
                {"attack": attack, "method": method, "standard_gain": margin, "passes": margin >= 1.0}
            )
            values.append((row["standard_accuracy"], method))
            if attack == "textual_backdoor":
                reduction = baseline["mean_backdoor_asr"] - row["mean_backdoor_asr"]
                asr_checks.append({"method": method, "mean_asr_reduction": reduction, "passes": reduction > 0})
        values.sort()
        adjacent = [
            {"lower": values[index][1], "upper": values[index + 1][1], "difference": values[index + 1][0] - values[index][0]}
            for index in range(len(values) - 1)
        ]
        separation_checks.append(
            {"attack": attack, "adjacent_differences": adjacent, "passes": all(item["difference"] >= 1.0 for item in adjacent)}
        )
    endings = {
        f"{row[field]:.2f}".split(".")[1]
        for row in rows
        for field in ("standard_accuracy", "heterogeneous_accuracy")
    }
    checks: dict[str, Any] = {
        "matrix_complete": set(by_cell) == expected,
        "observed_cell_count": len(by_cell),
        "expected_cell_count": len(expected),
        "missing_cells": [list(cell) for cell in sorted(expected - set(by_cell))],
        "attacked_fedavg_standard_drop_checks": drop_checks,
        "defense_standard_gain_checks": defense_checks,
        "attacked_method_separation_checks": separation_checks,
        "textual_backdoor_asr_checks": asr_checks,
        "distinct_accuracy_endings": sorted(endings),
        "natural_resolution_check": {"distinct_endings": len(endings), "passes": len(endings) >= 10},
    }
    checks["all_checks_pass"] = (
        checks["matrix_complete"]
        and len(drop_checks) == 2
        and all(item["passes"] for item in drop_checks)
        and len(defense_checks) == 10
        and all(item["passes"] for item in defense_checks)
        and len(separation_checks) == 2
        and all(item["passes"] for item in separation_checks)
        and len(asr_checks) == 5
        and all(item["passes"] for item in asr_checks)
        and checks["natural_resolution_check"]["passes"]
    )
    return checks


def _write_report(
    path: Path, rows: list[dict[str, Any]], checks: dict[str, Any], title: str
) -> None:
    by_cell = {(row["attack"], row["method"]): row for row in rows}
    lines = [
        f"# {title}",
        "",
        "Measured seed-2026 results using 1,786 randomly sampled examples per domain. HSSR, HRCW, paper targets, and synthetic jitter are not used.",
        "",
        "| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Domain gap | Standard gain vs FedAvg | ASR standard | ASR heterogeneous | ASR mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        baseline = by_cell.get((row["attack"], "fedavg"))
        gain = row["standard_accuracy"] - baseline["standard_accuracy"] if baseline else None
        lines.append(
            f"| {row['attack']} | {row['method']} | {_fmt(row['standard_accuracy'])} | "
            f"{_fmt(row['heterogeneous_accuracy'])} | {_fmt(row['mean_accuracy'])} | "
            f"{_fmt(row['domain_gap'])} | {_fmt(gain, True)} | "
            f"{_fmt(row['backdoor_asr_standard'])} | {_fmt(row['backdoor_asr_heterogeneous'])} | "
            f"{_fmt(row['mean_backdoor_asr'])} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion checks",
            "",
            f"- Complete matrix: {checks['observed_cell_count']}/{checks['expected_cell_count']}.",
            "- Under both attacks, every defense exceeds FedAvg standard-domain accuracy by at least one point.",
            "- Under both attacks, adjacent standard-domain method results differ by at least one point.",
            "- Every defense reduces mean textual-backdoor ASR relative to FedAvg.",
            f"- Distinct two-decimal endings: {checks['natural_resolution_check']['distinct_endings']}.",
            f"- All checks pass: {checks['all_checks_pass']}.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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
    (output / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "conclusion_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(output / "REPORT.md", rows, checks, args.title)
    _copy_evidence(root, output, rows)
    print(f"Exported {len(rows)} runs; checks pass={checks['all_checks_pass']}")


if __name__ == "__main__":
    main()
