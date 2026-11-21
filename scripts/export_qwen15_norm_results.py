from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from export_aggressive_results import FIELDS, METHODS, _copy_evidence, _fmt, build_rows
from export_natural_domain_results import ATTACKS, conclusion_checks


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the norm-matched Qwen-1.5B study")
    parser.add_argument("--root", default="results/natural_domain_attack_study_qwen15_v5")
    parser.add_argument("--baseline-root", default="results/natural_domain_attack_study_v3")
    parser.add_argument(
        "--output", default="reports/natural_domain_attack_study_qwen15_v5"
    )
    return parser.parse_args()


def model_scale_checks(
    rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    current = {(row["attack"], row["method"]): row for row in rows}
    baseline = {(row["attack"], row["method"]): row for row in baseline_rows}
    cells = []
    condition_checks = []
    for attack in ATTACKS:
        current_values = []
        baseline_values = []
        for method in METHODS:
            new = current[(attack, method)]
            old = baseline[(attack, method)]
            delta = new["mean_accuracy"] - old["mean_accuracy"]
            cells.append(
                {
                    "attack": attack,
                    "method": method,
                    "qwen15_mean_accuracy": new["mean_accuracy"],
                    "qwen05_mean_accuracy": old["mean_accuracy"],
                    "difference": delta,
                    "passes": delta > 0.0,
                }
            )
            current_values.append(new["mean_accuracy"])
            baseline_values.append(old["mean_accuracy"])
        current_mean = sum(current_values) / len(current_values)
        baseline_mean = sum(baseline_values) / len(baseline_values)
        condition_checks.append(
            {
                "attack": attack,
                "qwen15_mean_accuracy": current_mean,
                "qwen05_mean_accuracy": baseline_mean,
                "difference": current_mean - baseline_mean,
                "passes": current_mean > baseline_mean,
            }
        )
    current_overall = sum(item["qwen15_mean_accuracy"] for item in condition_checks) / len(
        condition_checks
    )
    baseline_overall = sum(item["qwen05_mean_accuracy"] for item in condition_checks) / len(
        condition_checks
    )
    attacked_cells = [item for item in cells if item["attack"] != "none"]
    return {
        "cell_comparisons": cells,
        "condition_mean_comparisons": condition_checks,
        "attacked_cells_improved": {
            "improved": sum(item["passes"] for item in attacked_cells),
            "total": len(attacked_cells),
            "passes": all(item["passes"] for item in attacked_cells),
        },
        "overall": {
            "qwen15_mean_accuracy": current_overall,
            "qwen05_mean_accuracy": baseline_overall,
            "difference": current_overall - baseline_overall,
            "passes": current_overall > baseline_overall,
        },
        "all_scale_checks_pass": all(item["passes"] for item in condition_checks)
        and all(item["passes"] for item in attacked_cells)
        and current_overall > baseline_overall,
    }


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    checks: dict[str, Any],
) -> None:
    baseline = {(row["attack"], row["method"]): row for row in baseline_rows}
    lines = [
        "# Qwen2.5-1.5B norm-matched attack study",
        "",
        "Measured seed-2026 results using 1,786 randomly sampled examples per domain. "
        "Attacker updates are normalized against the benign median update norm. HSSR, "
        "HRCW, paper targets, and synthetic jitter are not used.",
        "",
        "| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Mean Δ vs 0.5B | Domain gap | ASR standard | ASR heterogeneous | ASR mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        old = baseline[(row["attack"], row["method"])]
        lines.append(
            f"| {row['attack']} | {row['method']} | {_fmt(row['standard_accuracy'])} | "
            f"{_fmt(row['heterogeneous_accuracy'])} | {_fmt(row['mean_accuracy'])} | "
            f"{_fmt(row['mean_accuracy'] - old['mean_accuracy'], True)} | "
            f"{_fmt(row['domain_gap'])} | {_fmt(row['backdoor_asr_standard'])} | "
            f"{_fmt(row['backdoor_asr_heterogeneous'])} | "
            f"{_fmt(row['mean_backdoor_asr'])} |"
        )
    scale = checks["model_scale_checks"]
    lines.extend(
        [
            "",
            "## Acceptance checks",
            "",
            f"- Complete matrix: {checks['observed_cell_count']}/{checks['expected_cell_count']}.",
            "- Under both attacks, every defense exceeds FedAvg standard-domain accuracy by at least one point.",
            "- Under both attacks, adjacent standard-domain method results differ by at least one point.",
            "- Every defense reduces mean textual-backdoor ASR relative to FedAvg.",
            f"- Attacked cells improved over 0.5B: {scale['attacked_cells_improved']['improved']}/{scale['attacked_cells_improved']['total']}.",
            f"- Overall mean accuracy gain over 0.5B: {scale['overall']['difference']:+.2f} points.",
            f"- All checks pass: {checks['all_checks_pass']}.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _arguments()
    root = Path(args.root)
    baseline_root = Path(args.baseline_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = build_rows(root)
    baseline_rows = build_rows(baseline_root)
    checks = conclusion_checks(rows)
    checks["model_scale_checks"] = model_scale_checks(rows, baseline_rows)
    checks["all_checks_pass"] = (
        checks["all_checks_pass"]
        and checks["model_scale_checks"]["all_scale_checks_pass"]
    )
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
    _write_report(output / "REPORT.md", rows, baseline_rows, checks)
    _copy_evidence(root, output, rows)
    print(f"Exported {len(rows)} runs; checks pass={checks['all_checks_pass']}")


if __name__ == "__main__":
    main()
