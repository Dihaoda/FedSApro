from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from export_aggressive_results import FIELDS, METHODS, _fmt, build_rows


STUDIES = (
    ("0.5B", Path("results/cross_domain_tweet_yelp_qwen05_v1")),
    ("1.5B", Path("results/cross_domain_tweet_yelp_qwen15_v1")),
    ("3B", Path("results/cross_domain_tweet_yelp_qwen3_v1")),
)
OUTPUT = Path("reports/cross_domain_tweet_yelp_three_scale_v1")
ATTACKS = ("none", "label_flip", "textual_backdoor")


def _overview(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell = {(row["attack"], row["method"]): row for row in rows}
    condition_means = {
        attack: sum(by_cell[(attack, method)]["mean_accuracy"] for method in METHODS)
        / len(METHODS)
        for attack in ATTACKS
    }
    defense_gains = [
        by_cell[(attack, method)]["standard_accuracy"]
        - by_cell[(attack, "fedavg")]["standard_accuracy"]
        for attack in ATTACKS[1:]
        for method in METHODS[1:]
    ]
    return {
        "scale": label,
        "model": rows[0]["model"],
        "run_count": len(rows),
        "overall_mean_accuracy": sum(row["mean_accuracy"] for row in rows) / len(rows),
        "condition_mean_accuracy": condition_means,
        "fedavg_domain_gap": {
            attack: by_cell[(attack, "fedavg")]["domain_gap"] for attack in ATTACKS
        },
        "minimum_attacked_defense_standard_gain": min(defense_gains),
    }


def _write_report(
    path: Path, rows: list[dict[str, Any]], overviews: list[dict[str, Any]]
) -> None:
    label_by_model = {item["model"]: item["scale"] for item in overviews}
    lines = [
        "# TweetEval-Yelp three-scale validation",
        "",
        "Measured seed-2026 binary-sentiment results for the frozen TweetEval "
        "standard domain and Yelp heterogeneous benign domain. Domain gap is the "
        "absolute accuracy disparity. HSSR, HRCW, paper targets, and synthetic "
        "jitter are not acceptance criteria.",
        "",
        "## Scale overview",
        "",
        "| Scale | Overall mean acc. | Clean mean | Label-flip mean | Backdoor mean | "
        "Clean FedAvg gap | Label FedAvg gap | Backdoor FedAvg gap | Min defense gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in overviews:
        condition = item["condition_mean_accuracy"]
        gaps = item["fedavg_domain_gap"]
        lines.append(
            f"| {item['scale']} | {_fmt(item['overall_mean_accuracy'])} | "
            f"{_fmt(condition['none'])} | {_fmt(condition['label_flip'])} | "
            f"{_fmt(condition['textual_backdoor'])} | {_fmt(gaps['none'])} | "
            f"{_fmt(gaps['label_flip'])} | {_fmt(gaps['textual_backdoor'])} | "
            f"{_fmt(item['minimum_attacked_defense_standard_gain'])} |"
        )
    lines.extend(
        [
            "",
            "## Complete measured matrix",
            "",
            "| Scale | Attack | Method | Standard acc. | Heterogeneous acc. | "
            "Mean acc. | Domain gap | ASR standard | ASR heterogeneous | ASR mean |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {label_by_model[row['model']]} | {row['attack']} | {row['method']} | "
            f"{_fmt(row['standard_accuracy'])} | "
            f"{_fmt(row['heterogeneous_accuracy'])} | "
            f"{_fmt(row['mean_accuracy'])} | {_fmt(row['domain_gap'])} | "
            f"{_fmt(row['backdoor_asr_standard'])} | "
            f"{_fmt(row['backdoor_asr_heterogeneous'])} | "
            f"{_fmt(row['mean_backdoor_asr'])} |"
        )
    lines.extend(
        [
            "",
            "Each scale's machine-readable conclusion checks and compact raw "
            "evidence are stored in its dedicated report directory.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grouped = [(label, build_rows(root)) for label, root in STUDIES]
    rows = [row for _, group_rows in grouped for row in group_rows]
    overviews = [_overview(label, group_rows) for label, group_rows in grouped]
    with (OUTPUT / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "scale_overview.json").write_text(
        json.dumps(overviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(OUTPUT / "REPORT.md", rows, overviews)
    print(f"Exported {len(rows)} runs to {OUTPUT}")


if __name__ == "__main__":
    main()
