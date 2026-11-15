from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .config import ATTACKS, METHODS, MODEL_KEYS, load_yaml, resolve_config
from .data import prepare_data, validate_data
from .diagnostics import summarize_run
from .runner import run_experiment


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/paper/base.yaml")
    parser.add_argument("--model", choices=MODEL_KEYS)
    parser.add_argument("--attack", choices=ATTACKS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted configuration override; VALUE is parsed as YAML",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedsapro")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-data", help="Create deterministic paper client splits")
    _common(prepare)
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="Run one model/attack/method/seed experiment")
    _common(run)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--prepare-if-missing", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Recompute a run summary from raw logs")
    summarize.add_argument("--run-dir", required=True)
    summarize.add_argument("--config", help="Defaults to RUN_DIR/resolved_config.yaml")

    validate = subparsers.add_parser("validate-data", help="Validate partition checksums and roles")
    _common(validate)
    return parser


def _resolved(args: argparse.Namespace) -> dict:
    overrides: dict[str, object] = {}
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"Invalid --set value {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    return resolve_config(
        args.config,
        model=args.model,
        attack=args.attack,
        method=args.method,
        seed=args.seed,
        smoke=args.smoke,
        overrides=overrides,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-data":
        root = prepare_data(_resolved(args), force=args.force)
        print(root)
        return
    if args.command == "validate-data":
        print(json.dumps(validate_data(_resolved(args)), ensure_ascii=False, indent=2))
        return
    if args.command == "run":
        config = _resolved(args)
        if args.prepare_if_missing:
            prepare_data(config)
        summary = run_experiment(config, resume=args.resume, force=args.force)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "summarize":
        run_dir = Path(args.run_dir)
        config_path = Path(args.config) if args.config else run_dir / "resolved_config.yaml"
        summary = summarize_run(load_yaml(config_path), run_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
