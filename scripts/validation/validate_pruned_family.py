#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open

PRUNABLE_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\."
    r"(?P<kind>experts\.gate_up_proj|experts\.down_proj|gate\.weight)$"
)


def routed_expert_count(config: dict[str, object]) -> int | None:
    candidates: list[dict[str, object]] = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        candidates.insert(0, text_config)
    for candidate in candidates:
        for key in ("num_experts", "n_routed_experts", "num_routed_experts"):
            value = candidate.get(key)
            if value is not None:
                return int(value)
    return None


def verify_tensor_headers(model: Path, *, keep: int, expected_layers: int) -> None:
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SystemExit(f"[fatal] invalid weight_map in {index_path}")

    matched: dict[tuple[int, str], str] = {}
    for key, shard_name in weight_map.items():
        match = PRUNABLE_RE.match(str(key))
        if not match:
            continue
        layer = int(match.group("layer"))
        kind = match.group("kind")
        matched[(layer, kind)] = str(shard_name)
        with safe_open(model / str(shard_name), framework="pt", device="cpu") as handle:
            shape = handle.get_slice(str(key)).get_shape()
        if not shape or int(shape[0]) != keep:
            raise SystemExit(f"[fatal] {key}: first axis={shape[0] if shape else None}, expected={keep}")

    expected = {
        (layer, kind)
        for layer in range(expected_layers)
        for kind in ("experts.gate_up_proj", "experts.down_proj", "gate.weight")
    }
    missing = sorted(expected - set(matched))
    if missing:
        raise SystemExit(f"[fatal] missing prunable tensors, first entries: {missing[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a 192/128/64 pruned checkpoint family.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--config-dir")
    parser.add_argument("--expected-criterion")
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument("--check-tensor-headers", action="store_true")
    args = parser.parse_args()

    model_root = Path(args.model_root)
    config_dir = Path(args.config_dir) if args.config_dir else None

    for keep in (192, 128, 64):
        if config_dir is not None:
            config_path = config_dir / f"run_config_keep{keep}.json"
            if not config_path.is_file():
                raise SystemExit(f"[fatal] missing evaluation config: {config_path}")
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
            model = Path(run_config["model"])
        else:
            model = model_root / f"keep{keep}"

        for name in ("config.json", "model.safetensors.index.json", "pruning_manifest.json"):
            if not (model / name).is_file():
                raise SystemExit(f"[fatal] missing {model / name}")

        manifest = json.loads((model / "pruning_manifest.json").read_text(encoding="utf-8"))
        if args.expected_criterion and args.expected_criterion not in json.dumps(manifest):
            raise SystemExit(
                f"[fatal] {model}: pruning manifest does not contain {args.expected_criterion!r}"
            )

        model_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
        observed = routed_expert_count(model_config)
        if observed != keep:
            raise SystemExit(f"[fatal] {model}: routed expert count={observed}, expected={keep}")

        if args.check_tensor_headers:
            verify_tensor_headers(model, keep=keep, expected_layers=args.expected_layers)

        print(f"[ok] keep{keep}: {model}")

    print(f"[ok] checkpoint family validated: {model_root}")


if __name__ == "__main__":
    main()
