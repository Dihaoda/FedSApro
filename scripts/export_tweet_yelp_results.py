from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from export_aggressive_results import FIELDS, _copy_evidence, _fmt, build_rows
from export_natural_domain_results import ATTACKS, conclusion_checks


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the TweetEval-Yelp study")
    parser.add_argument("--root", default="results/cross_domain_tweet_yelp_qwen05_v1")
    parser.add_argument("--output", default="reports/cross_domain_tweet_yelp_qwen05_v1")
    return parser.parse_args()


def _domain_gap_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell = {(row["attack"], row["method"]): row for row in rows}
    clean = by_cell.get(("none", "fedavg"))
    checks: list[dict[str, Any]] = []
    if clean is None:
        return checks
    for attack in ATTACKS[1:]:
        attacked = by_cell.get((attack, "fedavg"))
        if attacked is None:
            continue
        increase = attacked["domain_gap"] - clean["domain_gap"]
        checks.append(
            {
                "attack": attack,
                "clean_domain_gap": clean["domain_gap"],
                "attacked_domain_gap": attacked["domain_gap"],
                "domain_gap_increase": increase,
                "passes": increase > 0.0,
            }
        )
    return checks


def _strong_defense_margin_checks(
    checks: dict[str, Any], minimum: float = 3.0
) -> list[dict[str, Any]]:
    return [
        {
            **item,
            "required_standard_gain": minimum,
            "passes": float(item["standard_gain"]) >= minimum,
        }
        for item in checks["defense_standard_gain_checks"]
    ]


def _write_report(
    path: Path, rows: list[dict[str, Any]], checks: dict[str, Any]
) -> None:
    by_cell = {(row["attack"], row["method"]): row for row in rows}
    lines = [
        "# TweetEval-Yelp Qwen2.5-0.5B validation study",
        "",
        "Measured seed-2026 binary-sentiment results. The standard domain is "
        "TweetEval sentiment after removing neutral examples; the heterogeneous "
        "benign domain is Yelp Polarity. The evaluation set contains 1,786 balanced "
        "examples per domain, sampled from frozen length pools. HSSR, HRCW, paper "
        "targets, and synthetic jitter are not used.",
        "",
        "| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | "
        "Domain gap | Standard gain vs FedAvg | ASR standard | "
        "ASR heterogeneous | ASR mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        baseline = by_cell.get((row["attack"], "fedavg"))
        gain = row["standard_accuracy"] - baseline["standard_accuracy"]
        lines.append(
            f"| {row['attack']} | {row['method']} | "
            f"{_fmt(row['standard_accuracy'])} | "
            f"{_fmt(row['heterogeneous_accuracy'])} | "
            f"{_fmt(row['mean_accuracy'])} | {_fmt(row['domain_gap'])} | "
            f"{_fmt(gain, True)} | {_fmt(row['backdoor_asr_standard'])} | "
            f"{_fmt(row['backdoor_asr_heterogeneous'])} | "
            f"{_fmt(row['mean_backdoor_asr'])} |"
        )
    gap_passed = sum(item["passes"] for item in checks["domain_gap_checks"])
    lines.extend(
        [
            "",
            "## Frozen conclusion checks",
            "",
            f"- Complete matrix: {checks['observed_cell_count']}/"
            f"{checks['expected_cell_count']}.",
            "- Each attack lowers FedAvg standard-domain accuracy by at least five points.",
            f"- Attacks increasing the FedAvg domain gap: {gap_passed}/2.",
            "- Under both attacks, every defense exceeds FedAvg standard-domain "
            "accuracy by at least three points.",
            f"- Minimum observed attacked-defense standard-domain gain: "
            f"{checks['minimum_defense_standard_gain']:.2f} points.",
            "- Under both attacks, adjacent standard-domain method results differ "
            "by at least one point.",
            "- Every defense reduces mean textual-backdoor ASR relative to FedAvg.",
            f"- Distinct two-decimal endings: "
            f"{checks['natural_resolution_check']['distinct_endings']}.",
            f"- All checks pass: {checks['all_checks_pass']}.",
            "",
            "Exact unrounded values and per-check decisions are in "
            "`summary.json` and `conclusion_checks.json`. Compact run evidence is "
            "under `evidence/`.",
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
    checks["domain_gap_checks"] = _domain_gap_checks(rows)
    checks["strong_defense_margin_checks"] = _strong_defense_margin_checks(checks)
    checks["minimum_defense_standard_gain"] = min(
        item["standard_gain"] for item in checks["strong_defense_margin_checks"]
    )
    checks["all_checks_pass"] = (
        checks["all_checks_pass"]
        and len(checks["domain_gap_checks"]) == 2
        and all(item["passes"] for item in checks["domain_gap_checks"])
        and len(checks["strong_defense_margin_checks"]) == 10
        and all(item["passes"] for item in checks["strong_defense_margin_checks"])
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
    _write_report(output / "REPORT.md", rows, checks)
    _copy_evidence(root, output, rows)
    print(f"Exported {len(rows)} runs; checks pass={checks['all_checks_pass']}")


if __name__ == "__main__":
    main()
