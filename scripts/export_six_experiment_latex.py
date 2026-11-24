from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "six_experiment_latex_tables.tex"

STUDIES = (
    (
        "Qwen2.5-0.5B on Amazon Polarity and IMDb",
        "qwen05_amazon_imdb",
        ROOT / "reports" / "aggressive_security_study_v1" / "summary.csv",
    ),
    (
        "Qwen2.5-1.5B on Amazon Polarity and IMDb",
        "qwen15_amazon_imdb",
        ROOT
        / "reports"
        / "natural_domain_attack_study_qwen15_v5"
        / "summary.csv",
    ),
    (
        "Qwen2.5-3B on Amazon Polarity and IMDb",
        "qwen3_amazon_imdb",
        ROOT / "reports" / "natural_domain_attack_study_qwen3_v1" / "summary.csv",
    ),
    (
        "Qwen2.5-0.5B on TweetEval and Yelp Polarity",
        "qwen05_tweet_yelp",
        ROOT / "reports" / "cross_domain_tweet_yelp_qwen05_v1" / "summary.csv",
    ),
    (
        "Qwen2.5-1.5B on TweetEval and Yelp Polarity",
        "qwen15_tweet_yelp",
        ROOT / "reports" / "cross_domain_tweet_yelp_qwen15_v1" / "summary.csv",
    ),
    (
        "Qwen2.5-3B on TweetEval and Yelp Polarity",
        "qwen3_tweet_yelp",
        ROOT / "reports" / "cross_domain_tweet_yelp_qwen3_v1" / "summary.csv",
    ),
)

ATTACKS = ("none", "label_flip", "textual_backdoor")
ATTACK_NAMES = {
    "none": "No Attack",
    "label_flip": "Label Flip",
    "textual_backdoor": "Textual Backdoor",
}
METHODS = (
    "fedavg",
    "indexguard",
    "flame",
    "fltrust",
    "safefedllm",
    "fedgraph",
)
METHOD_NAMES = {
    "fedavg": "FedAvg",
    "indexguard": "IndexGuard",
    "flame": "FLAME",
    "fltrust": "FLTrust",
    "safefedllm": "Safe-FedLLM",
    "fedgraph": "FedGraph",
}


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_cell = {(row["attack"], row["method"]): row for row in rows}
    expected = {(attack, method) for attack in ATTACKS for method in METHODS}
    missing = sorted(expected - set(by_cell))
    if missing:
        raise ValueError(f"Missing cells in {path}: {missing}")
    return by_cell


def latex_table(title: str, label: str, path: Path) -> list[str]:
    rows = read_rows(path)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{title}}}",
        rf"\label{{tab:{label}}}",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Attack & Method & Standard Domain & Heterogeneous Domain & Domain Gap \\",
        r"\midrule",
    ]
    for attack_index, attack in enumerate(ATTACKS):
        for method_index, method in enumerate(METHODS):
            row = rows[(attack, method)]
            attack_name = (
                rf"\multirow{{{len(METHODS)}}}{{*}}{{{ATTACK_NAMES[attack]}}}"
                if method_index == 0
                else ""
            )
            lines.append(
                f"{attack_name} & {METHOD_NAMES[method]} & "
                f"{float(row['standard_accuracy']):.2f} & "
                f"{float(row['heterogeneous_accuracy']):.2f} & "
                f"{float(row['domain_gap']):.2f} \\\\"
            )
        if attack_index < len(ATTACKS) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return lines


def main() -> None:
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{multirow}",
        r"\usepackage{caption}",
        r"\captionsetup{font=small}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{document}",
        "",
    ]
    for index, (title, label, path) in enumerate(STUDIES):
        lines.extend(latex_table(title, label, path))
        if index < len(STUDIES) - 1:
            lines.extend([r"\clearpage", ""])
    lines.extend(["", r"\end{document}", ""])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported {len(STUDIES)} tables to {OUTPUT}")


if __name__ == "__main__":
    main()

