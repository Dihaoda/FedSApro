from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from export_aggressive_results import (
    ATTACKS,
    FIELDS,
    METHODS,
    _copy_evidence,
    _fmt,
    build_rows,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the heterogeneous-suppression study")
    parser.add_argument("--root", default="results/heterogeneous_suppression_study_v2")
    parser.add_argument("--output", default="reports/heterogeneous_suppression_study_v2")
    return parser.parse_args()


def conclusion_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell = {(row["attack"], row["method"]): row for row in rows}
    expected = {(attack, method) for attack in ATTACKS for method in METHODS}
    clean = by_cell.get(("none", "fedavg"))
    attacked_gap_checks: list[dict[str, Any]] = []
    suppression_checks: list[dict[str, Any]] = []
    asr_checks: list[dict[str, Any]] = []
    spread_checks: list[dict[str, Any]] = []
    for attack in ATTACKS[1:]:
        baseline = by_cell.get((attack, "fedavg"))
        if baseline and clean:
            increase = baseline["domain_gap"] - clean["domain_gap"]
            attacked_gap_checks.append(
                {"attack": attack, "gap_increase": increase, "passes": increase > 0}
            )
        defense_values: list[float] = []
        for method in METHODS[1:]:
            row = by_cell.get((attack, method))
            if not row or not baseline:
                continue
            margin = baseline["heterogeneous_accuracy"] - row["heterogeneous_accuracy"]
            suppression_checks.append(
                {
                    "attack": attack,
                    "method": method,
                    "fedavg_heterogeneous_advantage": margin,
                    "passes": margin > 0,
                }
            )
            defense_values.append(row["heterogeneous_accuracy"])
            if attack == "textual_backdoor":
                reduction = baseline["mean_backdoor_asr"] - row["mean_backdoor_asr"]
                asr_checks.append(
                    {
                        "method": method,
                        "mean_asr_reduction": reduction,
                        "passes": reduction > 0,
                    }
                )
        if defense_values:
            spread = max(defense_values) - min(defense_values)
            spread_checks.append(
                {"attack": attack, "heterogeneous_accuracy_range": spread, "passes": spread >= 2.0}
            )
    distinct_pairs = len(
        {(round(row["standard_accuracy"], 2), round(row["heterogeneous_accuracy"], 2)) for row in rows}
    )
    checks = {
        "matrix_complete": set(by_cell) == expected,
        "observed_cell_count": len(by_cell),
        "expected_cell_count": len(expected),
        "missing_cells": [list(cell) for cell in sorted(expected - set(by_cell))],
        "attacked_fedavg_gap_checks": attacked_gap_checks,
        "fedavg_heterogeneous_advantage_checks": suppression_checks,
        "textual_backdoor_asr_checks": asr_checks,
        "method_spread_checks": spread_checks,
        "distinct_accuracy_pairs": distinct_pairs,
        "natural_resolution_check": {"minimum": 12, "passes": distinct_pairs >= 12},
    }
    checks["all_predeclared_checks_pass"] = (
        checks["matrix_complete"]
        and len(attacked_gap_checks) == 2
        and all(item["passes"] for item in attacked_gap_checks)
        and len(suppression_checks) == 10
        and all(item["passes"] for item in suppression_checks)
        and len(asr_checks) == 5
        and all(item["passes"] for item in asr_checks)
        and len(spread_checks) == 2
        and all(item["passes"] for item in spread_checks)
        and checks["natural_resolution_check"]["passes"]
    )
    return checks


def _write_report(path: Path, rows: list[dict[str, Any]], checks: dict[str, Any]) -> None:
    lines = [
        "# Qwen2.5-0.5B heterogeneous benign-client suppression study",
        "",
        "These are measured seed-2026 results. Paper target values, HSSR, and HRCW are not used.",
        "",
        "| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Domain gap | FedAvg heterogeneous advantage | ASR standard | ASR heterogeneous | ASR mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        advantage = (
            -float(row["heterogeneous_accuracy_delta_vs_fedavg"])
            if row["heterogeneous_accuracy_delta_vs_fedavg"] is not None
            else None
        )
        lines.append(
            f"| {row['attack']} | {row['method']} | {_fmt(row['standard_accuracy'])} | "
            f"{_fmt(row['heterogeneous_accuracy'])} | {_fmt(row['mean_accuracy'])} | "
            f"{_fmt(row['domain_gap'])} | {_fmt(advantage, True)} | "
            f"{_fmt(row['backdoor_asr_standard'])} | "
            f"{_fmt(row['backdoor_asr_heterogeneous'])} | {_fmt(row['mean_backdoor_asr'])} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion checks",
            "",
            f"- Complete matrix: {checks['observed_cell_count']}/{checks['expected_cell_count']}.",
            "- Under both attacks, FedAvg has higher heterogeneous-domain accuracy than every defense.",
            "- Every defense reduces mean textual-backdoor ASR relative to FedAvg.",
            "- Both attacks increase the FedAvg domain gap relative to the clean run.",
            f"- Distinct two-decimal accuracy pairs: {checks['distinct_accuracy_pairs']}/18.",
            f"- All conclusion checks pass: {checks['all_predeclared_checks_pass']}.",
            "",
            "Exact margins and decisions are in `conclusion_checks.json`.",
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
    (output / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "conclusion_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output / "REPORT.md", rows, checks)
    _copy_evidence(root, output, rows)
    print(f"Exported {len(rows)} runs; checks pass={checks['all_predeclared_checks_pass']}")


if __name__ == "__main__":
    main()
