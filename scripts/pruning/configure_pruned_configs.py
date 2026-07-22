#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_model_spec(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--model must have the form KEEP_SIZE=/path/to/checkpoint"
        )
    keep_text, path_text = value.split("=", 1)
    try:
        keep = int(keep_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid keep size: {keep_text}") from exc
    if keep <= 0:
        raise argparse.ArgumentTypeError("keep size must be positive")
    path = Path(path_text).expanduser().resolve()
    return keep, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create one or more model-specific run configs from the exact "
            "Option-B baseline config. Selective generation is supported so a "
            "selective rebuilds can update only requested checkpoint configs."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="KEEP=PATH",
        help="Repeat for each checkpoint to configure, e.g. --model 128=/path/model.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for run_config_keep<KEEP>.json files.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/run_config.json",
        help="Exact evaluation-contract config used as the template.",
    )
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="Do not require pruning_manifest.json (not recommended).",
    )
    args = parser.parse_args()

    base_path = Path(args.base_config)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs: dict[int, Path] = {}
    for raw in args.model:
        keep, model = parse_model_spec(raw)
        if keep in specs:
            raise SystemExit(f"duplicate keep size: {keep}")
        if not model.is_dir():
            raise SystemExit(f"missing model directory: {model}")
        if not (model / "config.json").is_file():
            raise SystemExit(f"missing config.json: {model}")
        if not args.allow_missing_manifest and not (model / "pruning_manifest.json").is_file():
            raise SystemExit(f"missing pruning_manifest.json: {model}")
        specs[keep] = model

    for keep in sorted(specs, reverse=True):
        model = specs[keep]
        cfg = dict(base)
        cfg["model"] = str(model)
        cfg["tokenizer"] = str(model)
        target = output_dir / f"run_config_keep{keep}.json"
        target.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print("[done]", target, model)


if __name__ == "__main__":
    main()
