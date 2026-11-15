from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROBUST_METHODS = ("indexguard", "flame", "fltrust")
FIELDS = (
    "model",
    "attack",
    "method",
    "seed",
    "final_round",
    "hssr",
    "hrcw",
    "standard_accuracy",
    "heterogeneous_accuracy",
    "domain_gap",
    "target_hssr",
    "target_standard_accuracy",
    "target_heterogeneous_accuracy",
    "accepted",
    "github_commit",
    "run_dir",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export auditable FedSApro matrix results")
    parser.add_argument("--root", default="results/reproduction_v1")
    parser.add_argument("--output", default="reports/reproduction_v1")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _row(path: Path, root: Path) -> dict[str, Any]:
    summary = _read_json(path)
    measured = summary["measured"]
    target = summary["target"]
    provenance_path = path.parent / "provenance.json"
    provenance = _read_json(provenance_path) if provenance_path.exists() else {}
    return {
        "model": summary["model"],
        "attack": summary["attack"],
        "method": summary["method"],
        "seed": summary["seed"],
        "final_round": summary["final_round"],
        "hssr": measured["hssr"],
        "hrcw": measured["hrcw"],
        "standard_accuracy": measured["standard_accuracy"],
        "heterogeneous_accuracy": measured["heterogeneous_accuracy"],
        "domain_gap": measured["domain_gap"],
        "target_hssr": target["hssr"],
        "target_standard_accuracy": target["standard_accuracy"],
        "target_heterogeneous_accuracy": target["heterogeneous_accuracy"],
        "accepted": summary["accepted"],
        "github_commit": provenance.get("github_commit"),
        "run_dir": str(path.parent.relative_to(root)),
    }


def _conclusion_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell = {
        (str(row["model"]), str(row["attack"]), str(row["method"])): row for row in rows
    }
    checks: list[dict[str, Any]] = []
    groups = sorted({(str(row["model"]), str(row["attack"])) for row in rows})
    for model, attack in groups:
        baseline = by_cell.get((model, attack, "fedavg"))
        if baseline is None:
            continue
        for method in ROBUST_METHODS:
            robust = by_cell.get((model, attack, method))
            if robust is None:
                continue
            suppression_increased = float(robust["hssr"]) > float(baseline["hssr"])
            heterogeneous_declined = float(robust["heterogeneous_accuracy"]) < float(
                baseline["heterogeneous_accuracy"]
            )
            standard_retained = float(robust["standard_accuracy"]) >= float(
                baseline["standard_accuracy"]
            ) - 5.0
            checks.append(
                {
                    "model": model,
                    "attack": attack,
                    "method": method,
                    "suppression_increased": suppression_increased,
                    "heterogeneous_accuracy_declined": heterogeneous_declined,
                    "standard_accuracy_retained_within_5_points": standard_retained,
                    "conclusion_consistent": (
                        suppression_increased and heterogeneous_declined and standard_retained
                    ),
                }
            )
    return checks


def _write_markdown(
    output: Path, rows: list[dict[str, Any]], checks: list[dict[str, Any]]
) -> None:
    accepted = sum(bool(row["accepted"]) for row in rows)
    lines = [
        "# FedSApro single-seed reproduction",
        "",
        "Measured values are generated from raw run logs; paper targets remain separate columns.",
        "",
        f"Strict target-window acceptance: {accepted}/{len(rows)} cells.",
        "",
        "Values are `measured (paper target)` in percentage points.",
        "",
        "| Model | Attack | Method | HSSR | Standard acc. | Heterogeneous acc. | Accepted |",
        "|---|---|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['attack']} | {row['method']} | "
            f"{float(row['hssr']):.2f} ({float(row['target_hssr']):.2f}) | "
            f"{float(row['standard_accuracy']):.2f} "
            f"({float(row['target_standard_accuracy']):.2f}) | "
            f"{float(row['heterogeneous_accuracy']):.2f} "
            f"({float(row['target_heterogeneous_accuracy']):.2f}) | "
            f"{'yes' if row['accepted'] else 'no'} |"
        )
    suppression_supported = sum(bool(item["suppression_increased"]) for item in checks)
    downstream_supported = sum(
        bool(item["suppression_increased"])
        and bool(item["heterogeneous_accuracy_declined"])
        for item in checks
    )
    standard_retained = sum(
        bool(item["standard_accuracy_retained_within_5_points"]) for item in checks
    )
    passed = sum(bool(item["conclusion_consistent"]) for item in checks)
    lines.extend(
        [
            "",
            "## Qualitative conclusion audit",
            "",
            f"{suppression_supported}/{len(checks)} robust-method comparisons show higher "
            "HSSR than the paired FedAvg baseline (mechanism-level support).",
            "",
            f"{standard_retained}/{len(checks)} retain standard-domain accuracy within five "
            "percentage points of the paired FedAvg baseline.",
            "",
            f"{downstream_supported}/{len(checks)} show both higher HSSR and lower "
            "heterogeneous-domain accuracy (downstream-impact support).",
            "",
            f"{passed}/{len(checks)} robust-method comparisons satisfy all three predeclared "
            "checks: higher HSSR than FedAvg, lower heterogeneous-domain accuracy, and "
            "standard-domain accuracy retained within five points.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _arguments()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        (_row(path, root) for path in root.rglob("summary.json")),
        key=lambda row: (str(row["model"]), str(row["attack"]), str(row["method"])),
    )
    checks = _conclusion_checks(rows)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "conclusion_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(output / "REPORT.md", rows, checks)
    print(f"Exported {len(rows)} runs and {len(checks)} conclusion checks to {output}")


if __name__ == "__main__":
    main()
