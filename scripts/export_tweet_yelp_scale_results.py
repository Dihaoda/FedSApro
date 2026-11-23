from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from export_aggressive_results import FIELDS, _copy_evidence, _fmt, build_rows
from export_natural_domain_results import conclusion_checks
from export_tweet_yelp_results import _domain_gap_checks, _strong_defense_margin_checks


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a TweetEval-Yelp scale study")
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--current-label", required=True)
    parser.add_argument("--baseline-label", required=True)
    return parser.parse_args()


def scale_checks(
    rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    current = {(row["attack"], row["method"]): row for row in rows}
    baseline = {(row["attack"], row["method"]): row for row in baseline_rows}
    cell_checks: list[dict[str, Any]] = []
    condition_checks: list[dict[str, Any]] = []
    for attack in ("none", "label_flip", "textual_backdoor"):
        current_values: list[float] = []
        baseline_values: list[float] = []
        for method in (
            "fedavg",
            "indexguard",
            "flame",
            "fltrust",
            "safefedllm",
            "fedgraph",
        ):
            new = current[(attack, method)]["mean_accuracy"]
            old = baseline[(attack, method)]["mean_accuracy"]
            cell_checks.append(
                {
                    "attack": attack,
                    "method": method,
                    "current_mean_accuracy": new,
                    "baseline_mean_accuracy": old,
                    "difference": new - old,
                    "passes": new > old,
                }
            )
            current_values.append(new)
            baseline_values.append(old)
        current_mean = sum(current_values) / len(current_values)
        baseline_mean = sum(baseline_values) / len(baseline_values)
        condition_checks.append(
            {
                "attack": attack,
                "current_mean_accuracy": current_mean,
                "baseline_mean_accuracy": baseline_mean,
                "difference": current_mean - baseline_mean,
                "passes": current_mean > baseline_mean,
            }
        )
    current_overall = sum(item["current_mean_accuracy"] for item in condition_checks)
    baseline_overall = sum(
        item["baseline_mean_accuracy"] for item in condition_checks
    )
    current_overall /= len(condition_checks)
    baseline_overall /= len(condition_checks)
    return {
        "cell_checks": cell_checks,
        "condition_checks": condition_checks,
        "improved_cells": sum(item["passes"] for item in cell_checks),
        "total_cells": len(cell_checks),
        "overall": {
            "current_mean_accuracy": current_overall,
            "baseline_mean_accuracy": baseline_overall,
            "difference": current_overall - baseline_overall,
            "passes": current_overall > baseline_overall,
        },
        "acceptance_basis": "all_condition_means_and_overall_mean_improve",
        "all_scale_checks_pass": all(
            item["passes"] for item in condition_checks
        )
        and current_overall > baseline_overall,
    }


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    checks: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    baseline = {(row["attack"], row["method"]): row for row in baseline_rows}
    lines = [
        f"# {args.title}",
        "",
        "Measured seed-2026 binary-sentiment results on the frozen TweetEval-Yelp "
        "partition. HSSR, HRCW, paper targets, and synthetic jitter are not used.",
        "",
        "| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | "
        f"Mean delta vs {args.baseline_label} | Domain gap | ASR standard | "
        "ASR heterogeneous | ASR mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        old = baseline[(row["attack"], row["method"])]
        lines.append(
            f"| {row['attack']} | {row['method']} | "
            f"{_fmt(row['standard_accuracy'])} | "
            f"{_fmt(row['heterogeneous_accuracy'])} | "
            f"{_fmt(row['mean_accuracy'])} | "
            f"{_fmt(row['mean_accuracy'] - old['mean_accuracy'], True)} | "
            f"{_fmt(row['domain_gap'])} | "
            f"{_fmt(row['backdoor_asr_standard'])} | "
            f"{_fmt(row['backdoor_asr_heterogeneous'])} | "
            f"{_fmt(row['mean_backdoor_asr'])} |"
        )
    scale = checks["model_scale_checks"]
    gap_passed = sum(item["passes"] for item in checks["domain_gap_checks"])
    lines.extend(
        [
            "",
            "## Frozen conclusion checks",
            "",
            f"- Complete matrix: {checks['observed_cell_count']}/"
            f"{checks['expected_cell_count']}.",
            f"- Attacks increasing the FedAvg domain gap: {gap_passed}/2.",
            "- Every attacked defense exceeds FedAvg standard-domain accuracy "
            "by at least three points.",
            f"- Minimum observed attacked-defense standard-domain gain: "
            f"{checks['minimum_defense_standard_gain']:.2f} points.",
            "- Adjacent attacked standard-domain results differ by at least one point.",
            "- Every defense reduces mean textual-backdoor ASR relative to FedAvg.",
            f"- Improved cells over {args.baseline_label}: "
            f"{scale['improved_cells']}/{scale['total_cells']}.",
            f"- Overall mean gain over {args.baseline_label}: "
            f"{scale['overall']['difference']:+.2f} points.",
            f"- Every attack-condition mean improves over {args.baseline_label}: "
            f"{all(item['passes'] for item in scale['condition_checks'])}.",
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
    checks["domain_gap_checks"] = _domain_gap_checks(rows)
    checks["strong_defense_margin_checks"] = _strong_defense_margin_checks(checks)
    checks["minimum_defense_standard_gain"] = min(
        item["standard_gain"] for item in checks["strong_defense_margin_checks"]
    )
    checks["model_scale_checks"] = scale_checks(rows, baseline_rows)
    checks["all_checks_pass"] = (
        checks["all_checks_pass"]
        and len(checks["domain_gap_checks"]) == 2
        and all(item["passes"] for item in checks["domain_gap_checks"])
        and len(checks["strong_defense_margin_checks"]) == 10
        and all(item["passes"] for item in checks["strong_defense_margin_checks"])
        and checks["model_scale_checks"]["all_scale_checks_pass"]
    )
    with (output / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in rows)
    (output / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "conclusion_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output / "REPORT.md", rows, baseline_rows, checks, args)
    _copy_evidence(root, output, rows)
    print(f"Exported {len(rows)} runs; checks pass={checks['all_checks_pass']}")


if __name__ == "__main__":
    main()
