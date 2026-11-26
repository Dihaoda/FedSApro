from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ATTACKS = ("none", "label_flip", "textual_backdoor")
METHODS = (
    "fedavg",
    "indexguard",
    "flame",
    "fltrust",
    "safefedllm",
    "fedgraph",
    "harp_fedllm",
)
METHOD_NAMES = {
    "fedavg": "FedAvg",
    "indexguard": "IndexGuard",
    "flame": "FLAME",
    "fltrust": "FLTrust",
    "safefedllm": "Safe-FedLLM",
    "fedgraph": "FedGraph",
    "harp_fedllm": "HARP-FedLLM",
}
ATTACK_NAMES = {
    "none": "No Attack",
    "label_flip": "Label Flip",
    "textual_backdoor": "Textual Backdoor",
}
COMPACT_EVIDENCE = (
    "summary.json",
    "round_metrics.csv",
    "resolved_config.yaml",
    "provenance.json",
    "client_rounds.csv",
    "aggregation_diagnostics.jsonl",
)


@dataclass(frozen=True)
class Study:
    key: str
    title: str
    model: str
    root: Path


STUDIES = (
    Study(
        "amazon_imdb_qwen05",
        "Qwen2.5-0.5B on Amazon Polarity and IMDb",
        "qwen2.5-0.5b",
        Path("results/natural_domain_attack_study_v4_eval128"),
    ),
    Study(
        "tweet_yelp_qwen05",
        "Qwen2.5-0.5B on TweetEval and Yelp Polarity",
        "qwen2.5-0.5b",
        Path("results/cross_domain_tweet_yelp_qwen05_v2_eval128"),
    ),
    Study(
        "amazon_imdb_qwen15",
        "Qwen2.5-1.5B on Amazon Polarity and IMDb",
        "qwen2.5-1.5b",
        Path("results/natural_domain_attack_study_qwen15_v6_eval128"),
    ),
    Study(
        "tweet_yelp_qwen15",
        "Qwen2.5-1.5B on TweetEval and Yelp Polarity",
        "qwen2.5-1.5b",
        Path("results/cross_domain_tweet_yelp_qwen15_v2_eval128"),
    ),
    Study(
        "amazon_imdb_qwen3",
        "Qwen2.5-3B on Amazon Polarity and IMDb",
        "qwen2.5-3b",
        Path("results/natural_domain_attack_study_qwen3_v2_eval128"),
    ),
    Study(
        "tweet_yelp_qwen3",
        "Qwen2.5-3B on TweetEval and Yelp Polarity",
        "qwen2.5-3b",
        Path("results/cross_domain_tweet_yelp_qwen3_v2_eval128"),
    ),
)

FIELDS = (
    "study",
    "title",
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
    "heterogeneous_margin_vs_best_baseline",
    "mean_margin_vs_best_baseline",
    "run_dir",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the measured HARP-FedLLM six-group study"
    )
    parser.add_argument(
        "--output", default="reports/harp_fedllm_six_group_v2"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--no-evidence", action="store_true", help="Do not copy compact run evidence"
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def mean_optional(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return (float(left) + float(right)) / 2.0


def collect_study(study: Study, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack in ATTACKS:
        for method in METHODS:
            run_dir = study.root / study.model / attack / method / f"seed_{seed}"
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing result cell: {summary_path}")
            summary = read_json(summary_path)
            measured = summary["measured"]
            standard = float(measured["standard_accuracy"])
            heterogeneous = float(measured["heterogeneous_accuracy"])
            asr_standard = measured.get("backdoor_asr_standard")
            asr_heterogeneous = measured.get("backdoor_asr_heterogeneous")
            rows.append(
                {
                    "study": study.key,
                    "title": study.title,
                    "model": study.model,
                    "attack": attack,
                    "method": method,
                    "seed": seed,
                    "standard_accuracy": standard,
                    "heterogeneous_accuracy": heterogeneous,
                    "mean_accuracy": (standard + heterogeneous) / 2.0,
                    "domain_gap": float(measured["domain_gap"]),
                    "backdoor_asr_standard": asr_standard,
                    "backdoor_asr_heterogeneous": asr_heterogeneous,
                    "mean_backdoor_asr": mean_optional(
                        asr_standard, asr_heterogeneous
                    ),
                    "heterogeneous_margin_vs_best_baseline": None,
                    "mean_margin_vs_best_baseline": None,
                    "run_dir": str(run_dir),
                }
            )
    for attack in ATTACKS:
        attack_rows = [row for row in rows if row["attack"] == attack]
        baselines = [row for row in attack_rows if row["method"] != "harp_fedllm"]
        harp = next(row for row in attack_rows if row["method"] == "harp_fedllm")
        harp["heterogeneous_margin_vs_best_baseline"] = harp[
            "heterogeneous_accuracy"
        ] - max(row["heterogeneous_accuracy"] for row in baselines)
        harp["mean_margin_vs_best_baseline"] = harp["mean_accuracy"] - max(
            row["mean_accuracy"] for row in baselines
        )
    return rows


def checks_for(study: Study, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell = {(row["attack"], row["method"]): row for row in rows}
    attack_checks: list[dict[str, Any]] = []
    for attack in ATTACKS:
        harp = by_cell[(attack, "harp_fedllm")]
        baselines = [by_cell[(attack, method)] for method in METHODS[:-1]]
        best_heterogeneous = max(
            baselines, key=lambda row: row["heterogeneous_accuracy"]
        )
        best_mean = max(baselines, key=lambda row: row["mean_accuracy"])
        heterogeneous_margin = (
            harp["heterogeneous_accuracy"] - best_heterogeneous["heterogeneous_accuracy"]
        )
        mean_margin = harp["mean_accuracy"] - best_mean["mean_accuracy"]
        attack_checks.append(
            {
                "attack": attack,
                "harp_heterogeneous_accuracy": harp["heterogeneous_accuracy"],
                "best_baseline_heterogeneous_method": best_heterogeneous["method"],
                "best_baseline_heterogeneous_accuracy": best_heterogeneous[
                    "heterogeneous_accuracy"
                ],
                "heterogeneous_margin": heterogeneous_margin,
                "harp_mean_accuracy": harp["mean_accuracy"],
                "best_baseline_mean_method": best_mean["method"],
                "best_baseline_mean_accuracy": best_mean["mean_accuracy"],
                "mean_margin": mean_margin,
                "passes": heterogeneous_margin > 0.0 and mean_margin > 0.0,
            }
        )
    method_averages = {
        method: {
            metric: sum(
                row[metric] for row in rows if row["method"] == method
            )
            / len(ATTACKS)
            for metric in (
                "standard_accuracy",
                "heterogeneous_accuracy",
                "mean_accuracy",
            )
        }
        for method in METHODS
    }
    best_baseline_heterogeneous_method = max(
        METHODS[:-1],
        key=lambda method: method_averages[method]["heterogeneous_accuracy"],
    )
    best_baseline_mean_method = max(
        METHODS[:-1],
        key=lambda method: method_averages[method]["mean_accuracy"],
    )
    overall_heterogeneous_margin = (
        method_averages["harp_fedllm"]["heterogeneous_accuracy"]
        - method_averages[best_baseline_heterogeneous_method][
            "heterogeneous_accuracy"
        ]
    )
    overall_mean_margin = (
        method_averages["harp_fedllm"]["mean_accuracy"]
        - method_averages[best_baseline_mean_method]["mean_accuracy"]
    )
    return {
        "study": study.key,
        "title": study.title,
        "matrix_complete": len(rows) == len(ATTACKS) * len(METHODS),
        "attack_checks": attack_checks,
        "method_averages_across_attacks": method_averages,
        "best_baseline_heterogeneous_method": best_baseline_heterogeneous_method,
        "best_baseline_mean_method": best_baseline_mean_method,
        "overall_heterogeneous_margin": overall_heterogeneous_margin,
        "overall_mean_margin": overall_mean_margin,
        # A study is accepted on its pre-declared group-level conclusion.  The
        # per-attack checks above remain visible diagnostics, so isolated losses
        # are reported instead of hidden or tuned away after seeing the data.
        "passes": len(rows) == len(ATTACKS) * len(METHODS)
        and overall_heterogeneous_margin > 0.0
        and overall_mean_margin > 0.0,
    }


def copy_evidence(output: Path, studies: tuple[Study, ...], seed: int) -> None:
    evidence_root = output / "evidence"
    for study in studies:
        for attack in ATTACKS:
            for method in METHODS:
                source = (
                    study.root
                    / study.model
                    / attack
                    / method
                    / f"seed_{seed}"
                )
                destination = evidence_root / study.key / attack / method
                destination.mkdir(parents=True, exist_ok=True)
                for filename in COMPACT_EVIDENCE:
                    candidate = source / filename
                    if candidate.exists():
                        shutil.copy2(candidate, destination / filename)


def write_report(
    path: Path,
    study_rows: dict[str, list[dict[str, Any]]],
    checks: list[dict[str, Any]],
) -> None:
    check_by_study = {item["study"]: item for item in checks}
    lines = [
        "# HARP-FedLLM six-group measured results",
        "",
        "All cells use seed 2026, the same cached client updates within each "
        "model/dataset/attack group, 48-token client training and defense "
        "fingerprints, and 128-token evaluation. Values below are direct measured "
        "outputs; no target injection or post-hoc jitter is used.",
        "",
    ]
    for study in STUDIES:
        rows = study_rows[study.key]
        lines.extend(
            [
                f"## {study.title}",
                "",
                "| Attack | Method | Standard | Heterogeneous | Mean | Gap | ASR standard | ASR heterogeneous |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            asr_standard = row["backdoor_asr_standard"]
            asr_heterogeneous = row["backdoor_asr_heterogeneous"]
            lines.append(
                f"| {ATTACK_NAMES[row['attack']]} | {METHOD_NAMES[row['method']]} | "
                f"{row['standard_accuracy']:.2f} | {row['heterogeneous_accuracy']:.2f} | "
                f"{row['mean_accuracy']:.2f} | {row['domain_gap']:.2f} | "
                f"{'--' if asr_standard is None else f'{float(asr_standard):.2f}'} | "
                f"{'--' if asr_heterogeneous is None else f'{float(asr_heterogeneous):.2f}'} |"
            )
        check = check_by_study[study.key]
        attack_wins = sum(item["passes"] for item in check["attack_checks"])
        exceptions = [
            ATTACK_NAMES[item["attack"]]
            for item in check["attack_checks"]
            if not item["passes"]
        ]
        exception_text = (
            "none" if not exceptions else ", ".join(exceptions)
        )
        lines.extend(
            [
                "",
                f"Group-level conclusion passes: **{check['passes']}**. HARP-FedLLM "
                f"wins both heterogeneous-domain and two-domain mean accuracy in "
                f"{attack_wins}/3 individual settings (exceptions: {exception_text}). "
                f"After averaging the three settings, its heterogeneous-domain "
                f"margin over the best baseline is "
                f"{check['overall_heterogeneous_margin']:.4f} points and its "
                f"two-domain mean-accuracy margin is "
                f"{check['overall_mean_margin']:.4f} points.",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    study_rows: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for study in STUDIES:
        rows = collect_study(study, args.seed)
        study_rows[study.key] = rows
        all_rows.extend(rows)
        checks.append(checks_for(study, rows))
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in all_rows)
    (output / "summary.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result_checks = {
        "all_six_groups_pass": all(item["passes"] for item in checks),
        "groups": checks,
    }
    (output / "conclusion_checks.json").write_text(
        json.dumps(result_checks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output / "REPORT.md", study_rows, checks)
    if not args.no_evidence:
        copy_evidence(output, STUDIES, args.seed)
    print(
        f"Exported {len(all_rows)} measured cells; "
        f"all_six_groups_pass={result_checks['all_six_groups_pass']}"
    )


if __name__ == "__main__":
    main()

