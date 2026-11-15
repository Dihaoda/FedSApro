from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ATTACK_ORDER = {"none": 0, "label_flip": 1, "textual_backdoor": 2}
METHOD_ORDER = {
    "fedavg": 0,
    "fedprox": 1,
    "scaffold": 2,
    "indexguard": 3,
    "flame": 4,
    "fltrust": 5,
}
FIELDS = (
    "model",
    "attack",
    "method",
    "seed",
    "final_round",
    "standard_accuracy",
    "heterogeneous_accuracy",
    "mean_accuracy",
    "domain_gap",
    "backdoor_asr_standard",
    "backdoor_asr_heterogeneous",
    "mean_backdoor_asr",
    "mean_accuracy_delta_vs_fedavg",
    "domain_gap_delta_vs_fedavg",
    "mean_backdoor_asr_delta_vs_fedavg",
    "domain_gap_delta_vs_no_attack",
    "github_commit",
    "run_dir",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the accuracy-centric Qwen reproduction table"
    )
    parser.add_argument("--root", default="results/accuracy_study_v1")
    parser.add_argument("--output", default="reports/accuracy_study_v1")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_evidence_bundle(root: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("summary.json")):
        run_dir = summary_path.parent
        provenance_path = run_dir / "provenance.json"
        config_path = run_dir / "resolved_config.yaml"
        runs.append(
            {
                "run_dir": str(run_dir.relative_to(root)),
                "summary": _read_json(summary_path),
                "provenance": (
                    _read_json(provenance_path) if provenance_path.exists() else None
                ),
                "resolved_config_yaml": (
                    config_path.read_text(encoding="utf-8") if config_path.exists() else None
                ),
                "round_metrics": _read_csv(run_dir / "round_metrics.csv"),
                "client_rounds": _read_csv(run_dir / "client_rounds.csv"),
            }
        )
    return {
        "schema_version": 1,
        "source_root": str(root),
        "run_count": len(runs),
        "runs": runs,
    }


def _average_optional(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return (float(left) + float(right)) / 2.0


def _base_row(path: Path, root: Path) -> dict[str, Any]:
    summary = _read_json(path)
    measured = summary["measured"]
    provenance_path = path.parent / "provenance.json"
    provenance = _read_json(provenance_path) if provenance_path.exists() else {}
    standard = float(measured["standard_accuracy"])
    heterogeneous = float(measured["heterogeneous_accuracy"])
    asr_standard = measured.get("backdoor_asr_standard")
    asr_heterogeneous = measured.get("backdoor_asr_heterogeneous")
    return {
        "model": summary["model"],
        "attack": summary["attack"],
        "method": summary["method"],
        "seed": int(summary["seed"]),
        "final_round": int(summary["final_round"]),
        "standard_accuracy": standard,
        "heterogeneous_accuracy": heterogeneous,
        "mean_accuracy": (standard + heterogeneous) / 2.0,
        "domain_gap": float(measured["domain_gap"]),
        "backdoor_asr_standard": asr_standard,
        "backdoor_asr_heterogeneous": asr_heterogeneous,
        "mean_backdoor_asr": _average_optional(asr_standard, asr_heterogeneous),
        "github_commit": provenance.get("github_commit"),
        "run_dir": str(path.parent.relative_to(root)),
    }


def build_rows(root: Path) -> list[dict[str, Any]]:
    rows = [_base_row(path, root) for path in root.rglob("summary.json")]
    by_cell = {
        (str(row["model"]), str(row["attack"]), str(row["method"]), int(row["seed"])): row
        for row in rows
    }
    for row in rows:
        cell = (str(row["model"]), str(row["attack"]), "fedavg", int(row["seed"]))
        fedavg = by_cell.get(cell)
        no_attack = by_cell.get(
            (str(row["model"]), "none", str(row["method"]), int(row["seed"]))
        )
        row["mean_accuracy_delta_vs_fedavg"] = (
            float(row["mean_accuracy"]) - float(fedavg["mean_accuracy"])
            if fedavg is not None
            else None
        )
        row["domain_gap_delta_vs_fedavg"] = (
            float(row["domain_gap"]) - float(fedavg["domain_gap"])
            if fedavg is not None
            else None
        )
        row["mean_backdoor_asr_delta_vs_fedavg"] = (
            float(row["mean_backdoor_asr"]) - float(fedavg["mean_backdoor_asr"])
            if fedavg is not None
            and row["mean_backdoor_asr"] is not None
            and fedavg["mean_backdoor_asr"] is not None
            else None
        )
        row["domain_gap_delta_vs_no_attack"] = (
            float(row["domain_gap"]) - float(no_attack["domain_gap"])
            if no_attack is not None
            else None
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["model"]),
            ATTACK_ORDER.get(str(row["attack"]), 99),
            METHOD_ORDER.get(str(row["method"]), 99),
            int(row["seed"]),
        ),
    )


def conclusion_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed_cells = {
        (str(row["model"]), str(row["attack"]), str(row["method"]), int(row["seed"]))
        for row in rows
    }
    study_pairs = {
        (str(row["model"]), int(row["seed"]))
        for row in rows
    }
    expected_cells = {
        (model, attack, method, seed)
        for model, seed in study_pairs
        for attack in ATTACK_ORDER
        for method in METHOD_ORDER
    }
    missing_cells = sorted(expected_cells - observed_cells)
    unexpected_cells = sorted(observed_cells - expected_cells)

    checks: list[dict[str, Any]] = []
    for row in rows:
        if row["attack"] == "none" or row["method"] == "fedavg":
            continue
        utility_gain = float(row["mean_accuracy_delta_vs_fedavg"])
        asr_delta = row["mean_backdoor_asr_delta_vs_fedavg"]
        asr_improved = asr_delta is None or float(asr_delta) < 0.0
        checks.append(
            {
                "model": row["model"],
                "attack": row["attack"],
                "method": row["method"],
                "mean_accuracy_gain_vs_fedavg": utility_gain,
                "mean_backdoor_asr_change_vs_fedavg": asr_delta,
                "utility_better_than_fedavg": utility_gain > 0.0,
                "backdoor_asr_better_than_fedavg": asr_improved,
                "overall_better_than_fedavg": utility_gain > 0.0 and asr_improved,
            }
        )

    attack_gap_checks: list[dict[str, Any]] = []
    fedavg_rows = {
        (str(row["model"]), str(row["attack"]), int(row["seed"])): row
        for row in rows
        if row["method"] == "fedavg"
    }
    for (model, attack, seed), row in fedavg_rows.items():
        if attack == "none":
            continue
        clean = fedavg_rows.get((model, "none", seed))
        if clean is None:
            continue
        delta = float(row["domain_gap"]) - float(clean["domain_gap"])
        attack_gap_checks.append(
            {
                "model": model,
                "attack": attack,
                "seed": seed,
                "clean_domain_gap": clean["domain_gap"],
                "attacked_domain_gap": row["domain_gap"],
                "domain_gap_increase": delta,
                "attack_increases_domain_gap": delta > 0.0,
            }
        )
    return {
        "matrix_complete": not missing_cells and not unexpected_cells,
        "expected_cell_count": len(expected_cells),
        "observed_cell_count": len(observed_cells),
        "missing_cells": [list(cell) for cell in missing_cells],
        "unexpected_cells": [list(cell) for cell in unexpected_cells],
        "attack_gap_checks": attack_gap_checks,
        "defense_checks": checks,
        "all_attacks_increase_fedavg_gap": bool(attack_gap_checks)
        and all(item["attack_increases_domain_gap"] for item in attack_gap_checks),
        "all_attacked_defenses_better_than_fedavg": bool(checks)
        and all(item["overall_better_than_fedavg"] for item in checks),
    }


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None or value == "":
        return "-"
    number = float(value)
    return f"{number:+.2f}" if signed else f"{number:.2f}"


def _write_markdown(
    output: Path, rows: list[dict[str, Any]], checks: dict[str, Any]
) -> None:
    lines = [
        "# Qwen2.5-0.5B accuracy-centric reproduction",
        "",
        "This report uses only measured single-seed results from the present study. It does not "
        "compare against paper values and does not use HSSR or HRCW.",
        "",
        "Accuracy, domain gap, and ASR are percentage points. `Mean acc. delta` is relative to "
        "FedAvg under the same attack. `Gap delta (clean)` is relative to the same method without "
        "an attack. Negative ASR delta is better.",
        "",
        "| Attack | Method | Standard acc. | Heterogeneous acc. | Mean acc. | Domain gap | "
        "Mean acc. delta | Gap delta (clean) | ASR standard | ASR heterogeneous | ASR delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['attack']} | {row['method']} | "
            f"{_fmt(row['standard_accuracy'])} | {_fmt(row['heterogeneous_accuracy'])} | "
            f"{_fmt(row['mean_accuracy'])} | {_fmt(row['domain_gap'])} | "
            f"{_fmt(row['mean_accuracy_delta_vs_fedavg'], signed=True)} | "
            f"{_fmt(row['domain_gap_delta_vs_no_attack'], signed=True)} | "
            f"{_fmt(row['backdoor_asr_standard'])} | "
            f"{_fmt(row['backdoor_asr_heterogeneous'])} | "
            f"{_fmt(row['mean_backdoor_asr_delta_vs_fedavg'], signed=True)} |"
        )

    gap_checks = checks["attack_gap_checks"]
    defense_checks = checks["defense_checks"]
    gaps_passed = sum(item["attack_increases_domain_gap"] for item in gap_checks)
    defenses_passed = sum(item["overall_better_than_fedavg"] for item in defense_checks)
    lines.extend(
        [
            "",
            "## Predeclared conclusion checks",
            "",
            f"- Complete attack/method matrix: {checks['observed_cell_count']}/"
            f"{checks['expected_cell_count']} cells.",
            f"- Attacks increasing the paired FedAvg domain gap: {gaps_passed}/{len(gap_checks)}.",
            f"- Attacked defenses outperforming paired FedAvg on mean accuracy, and also on ASR "
            f"for textual backdoor: {defenses_passed}/{len(defense_checks)}.",
            "",
            "The machine-readable decisions and exact deltas are in `conclusion_checks.json`.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


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
    (output / "evidence_bundle.json").write_text(
        json.dumps(build_evidence_bundle(root), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output / "REPORT.md", rows, checks)
    print(f"Exported {len(rows)} measured runs to {output}")


if __name__ == "__main__":
    main()
