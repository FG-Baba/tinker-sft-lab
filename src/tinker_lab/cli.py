from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from tinker_lab.data import DataValidationError, load_examples
from tinker_lab.safety import LiveRunBlocked, require_live_run_authorization
from tinker_lab.training import DEFAULT_MODEL, SFTConfig, run_sft


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _doctor() -> tuple[dict[str, object], int]:
    key_present = bool(os.environ.get("TINKER_API_KEY"))
    tinker_version = _package_version("tinker")
    cookbook_version = _package_version("tinker-cookbook")
    python_supported = sys.version_info >= (3, 11)
    ready = all((python_supported, key_present, tinker_version, cookbook_version))
    return (
        {
            "ready": ready,
            "python": sys.version.split()[0],
            "python_3_11_or_newer": python_supported,
            "tinker_sdk": tinker_version,
            "tinker_cookbook": cookbook_version,
            "tinker_api_key_present": key_present,
        },
        0 if ready else 2,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinker-lab", description="Minimal Tinker SFT starter")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="Check local readiness without an API call")
    doctor.add_argument("--json", action="store_true")

    validate = commands.add_parser("validate", help="Validate a local JSONL dataset")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--json", action="store_true")

    train = commands.add_parser("train-sft", help="Run a small, billed SFT experiment")
    train.add_argument("dataset", type=Path)
    train.add_argument("--model", default=DEFAULT_MODEL)
    train.add_argument("--steps", type=int, default=1)
    train.add_argument("--max-examples", type=int, default=8)
    train.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize the billed call",
    )
    return parser


def _print(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        if args.command == "doctor":
            payload, exit_code = _doctor()
            _print(payload, as_json=args.json)
            return exit_code

        if args.command == "validate":
            examples = load_examples(args.dataset)
            _print(
                {"status": "valid", "dataset": str(args.dataset), "examples": len(examples)},
                as_json=args.json,
            )
            return 0

        require_live_run_authorization(
            execute=args.execute,
            api_key=os.environ.get("TINKER_API_KEY"),
        )
        examples = load_examples(args.dataset)
        config = SFTConfig(
            model=args.model,
            steps=args.steps,
            max_examples=args.max_examples,
        )
        result = asyncio.run(run_sft(examples, config, execute=args.execute))
        _print(result.as_dict(), as_json=True)
        return 0
    except (DataValidationError, LiveRunBlocked, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ImportError:
        print("Tinker dependencies are missing. Run: uv sync", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Tinker run failed ({type(exc).__name__}).", file=sys.stderr)
        return 1
