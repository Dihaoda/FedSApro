from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from fedsapro.config import ATTACKS, METHODS, MODEL_KEYS, experiment_run_dir, resolve_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a restart-safe FedSApro experiment matrix")
    parser.add_argument("--config", default="configs/reproduction/efficient.yaml")
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=["qwen2.5-1.5b"])
    parser.add_argument("--attacks", nargs="+", choices=ATTACKS, default=list(ATTACKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--try-seeds-until-accepted",
        action="store_true",
        help="Stop trying candidate seeds for a cell after its first accepted run",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed subprocess and continue with the next candidate",
    )
    parser.add_argument("--manifest", default="results/matrix_events.jsonl")
    return parser


def _parse_overrides(values: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --set value {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    return overrides


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = _parser().parse_args()
    overrides = _parse_overrides(args.overrides)
    manifest = Path(args.manifest)
    for model, attack, method in itertools.product(args.models, args.attacks, args.methods):
        for seed in args.seeds:
            config = resolve_config(
                args.config,
                model=model,
                attack=attack,
                method=method,
                seed=seed,
                overrides=overrides,
            )
            run_dir = experiment_run_dir(config)
            summary_path = run_dir / "summary.json"
            existing = _load_summary(summary_path)
            if existing and int(existing.get("final_round", -1)) == int(
                config["federated"]["rounds"]
            ):
                event = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "status": "already_complete",
                    "run_dir": str(run_dir),
                    "accepted": bool(existing.get("accepted")),
                }
                print(json.dumps(event, ensure_ascii=False), flush=True)
                _append_event(manifest, event)
                if args.try_seeds_until_accepted and existing.get("accepted"):
                    break
                continue

            command = [
                sys.executable,
                "-m",
                "fedsapro.cli",
                "run",
                "--config",
                args.config,
                "--model",
                model,
                "--attack",
                attack,
                "--method",
                method,
                "--seed",
                str(seed),
            ]
            command.append("--resume" if (run_dir / "checkpoint_latest.pt").exists() else "--force")
            for item in args.overrides:
                command.extend(["--set", item])

            started = datetime.now(timezone.utc).isoformat()
            print(f"[{started}] starting {model}/{attack}/{method}/seed_{seed}", flush=True)
            completed = subprocess.run(command, check=False)
            summary = _load_summary(summary_path)
            event = {
                "time": datetime.now(timezone.utc).isoformat(),
                "started": started,
                "status": "complete" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "run_dir": str(run_dir),
                "accepted": bool((summary or {}).get("accepted")),
            }
            print(json.dumps(event, ensure_ascii=False), flush=True)
            _append_event(manifest, event)
            if completed.returncode != 0 and not args.continue_on_error:
                return completed.returncode
            if args.try_seeds_until_accepted and (summary or {}).get("accepted"):
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
