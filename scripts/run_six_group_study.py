from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class Study:
    config: str
    model: str


STUDIES = {
    "amazon_imdb_qwen05": Study(
        "configs/reproduction/natural_domain_attack_study.yaml", "qwen2.5-0.5b"
    ),
    "amazon_imdb_qwen15": Study(
        "configs/reproduction/natural_domain_attack_study_qwen15.yaml", "qwen2.5-1.5b"
    ),
    "amazon_imdb_qwen3": Study(
        "configs/reproduction/natural_domain_attack_study_qwen3_v1.yaml", "qwen2.5-3b"
    ),
    "tweet_yelp_qwen05": Study(
        "configs/reproduction/cross_domain_tweet_yelp_qwen05_v1.yaml", "qwen2.5-0.5b"
    ),
    "tweet_yelp_qwen15": Study(
        "configs/reproduction/cross_domain_tweet_yelp_qwen15_v1.yaml", "qwen2.5-1.5b"
    ),
    "tweet_yelp_qwen3": Study(
        "configs/reproduction/cross_domain_tweet_yelp_qwen3_v1.yaml", "qwen2.5-3b"
    ),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen HARP-FedLLM six-group study")
    parser.add_argument(
        "--studies", nargs="+", choices=tuple(STUDIES), default=list(STUDIES)
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool) -> int:
    print(" ".join(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def main() -> int:
    args = arguments()
    for key in args.studies:
        study = STUDIES[key]
        if not Path(study.config).exists():
            raise FileNotFoundError(study.config)
        if args.prepare_data:
            code = run(
                [sys.executable, "-m", "fedsapro.cli", "prepare-data", "--config", study.config],
                dry_run=args.dry_run,
            )
            if code:
                return code
        manifest = f"results/manifests/{key}.jsonl"
        command = [
            sys.executable,
            "scripts/run_matrix.py",
            "--config",
            study.config,
            "--models",
            study.model,
            "--attacks",
            *ATTACKS,
            "--methods",
            *METHODS,
            "--seeds",
            str(args.seed),
            "--continue-on-error",
            "--manifest",
            manifest,
        ]
        code = run(command, dry_run=args.dry_run)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


