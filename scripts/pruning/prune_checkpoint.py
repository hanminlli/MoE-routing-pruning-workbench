#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"

# Only the 40 target-language-model MoE layers are calibrated by the GDPval
# response-token accounting. Qwen3.6 also ships a separate MTP draft layer
# under the ``mtp.`` prefix; it must not be mistaken for language layer 0.
PRUNABLE_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\."
    r"(?P<kind>experts\.gate_up_proj|experts\.down_proj|gate\.weight)$"
)

MTP_PREFIX = "mtp."


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_source_model(source_model: str, local_files_only: bool) -> Path:
    candidate = Path(source_model).expanduser()
    if candidate.exists():
        return candidate.resolve()

    print(f"[info] resolving Hugging Face snapshot: {source_model}")
    resolved = snapshot_download(
        repo_id=source_model,
        local_files_only=local_files_only,
    )
    return Path(resolved).resolve()


def parse_plan(
    plan_path: Path,
    *,
    expected_layers: int,
    expected_original_experts: int,
) -> tuple[dict[int, list[int]], int, dict[str, Any]]:
    plan = load_json(plan_path)
    layers_obj = plan.get("layers")
    if not isinstance(layers_obj, dict):
        raise ValueError("plan JSON is missing object field 'layers'")

    retained_by_layer: dict[int, list[int]] = {}
    keep_sizes: set[int] = set()

    for layer in range(expected_layers):
        info = layers_obj.get(str(layer))
        if not isinstance(info, dict):
            raise ValueError(f"plan is missing layer {layer}")

        ids = info.get("retained_original_expert_ids")
        if not isinstance(ids, list):
            raise ValueError(
                f"plan layer {layer} is missing retained_original_expert_ids"
            )

        retained = [int(x) for x in ids]
        if len(retained) != len(set(retained)):
            raise ValueError(f"duplicate retained expert ID in layer {layer}")
        if retained != sorted(retained):
            raise ValueError(
                f"layer {layer} retained_original_expert_ids must be sorted ascending"
            )
        if any(x < 0 or x >= expected_original_experts for x in retained):
            raise ValueError(f"out-of-range retained expert ID in layer {layer}")

        keep_sizes.add(len(retained))
        retained_by_layer[layer] = retained

    if len(keep_sizes) != 1:
        raise ValueError(f"plan uses inconsistent keep sizes: {sorted(keep_sizes)}")

    keep_size = next(iter(keep_sizes))
    declared = plan.get("retained_num_routed_experts_per_layer")
    if declared is not None and int(declared) != keep_size:
        raise ValueError(
            f"plan declares keep size {declared}, but layer lists contain {keep_size}"
        )

    return retained_by_layer, keep_size, plan


def get_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return config


def discover_prunable_keys(
    weight_map: dict[str, str],
    *,
    expected_layers: int,
) -> dict[str, tuple[int, str]]:
    found: dict[str, tuple[int, str]] = {}
    kinds_by_layer: dict[int, set[str]] = {i: set() for i in range(expected_layers)}

    for key in weight_map:
        match = PRUNABLE_RE.match(key)
        if match is None:
            continue
        layer = int(match.group("layer"))
        kind = match.group("kind")
        if layer < 0 or layer >= expected_layers:
            continue
        found[key] = (layer, kind)
        kinds_by_layer[layer].add(kind)

    expected_kinds = {
        "experts.gate_up_proj",
        "experts.down_proj",
        "gate.weight",
    }
    for layer in range(expected_layers):
        if kinds_by_layer[layer] != expected_kinds:
            raise ValueError(
                f"layer {layer} prunable tensor set is {sorted(kinds_by_layer[layer])}; "
                f"expected {sorted(expected_kinds)}"
            )

    expected_count = expected_layers * len(expected_kinds)
    if len(found) != expected_count:
        raise ValueError(
            f"found {len(found)} prunable tensors; expected {expected_count}"
        )
    return found


def is_mtp_key(key: str) -> bool:
    return key.startswith(MTP_PREFIX)


def copy_auxiliary_files(source_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for source_path in sorted(source_dir.iterdir()):
        if not source_path.is_file():
            continue
        if source_path.name == INDEX_NAME or source_path.suffix == ".safetensors":
            continue
        destination = output_dir / source_path.name
        shutil.copy2(source_path, destination, follow_symlinks=True)
        copied.append(source_path.name)
    return copied


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Physically prune Qwen3.5/Qwen3.6 routed experts by streaming its "
            "Safetensors shards and slicing the routed expert bank per layer."
        )
    )
    parser.add_argument(
        "--source-model",
        default="Qwen/Qwen3.6-35B-A3B",
        help="Hugging Face model ID or local snapshot/checkpoint directory.",
    )
    parser.add_argument("--plan", required=True, help="Pruning-plan JSON file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument("--expected-original-experts", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-mtp",
        action="store_true",
        help=(
            "Keep original MTP draft weights. Not recommended when num_experts is changed, "
            "because the MTP layer was not calibrated and retains the original expert shape."
        ),
    )
    args = parser.parse_args()

    source_dir = resolve_source_model(args.source_model, args.local_files_only)
    plan_path = Path(args.plan).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("output directory must not be the source checkpoint or inside it")
    if not plan_path.exists():
        raise FileNotFoundError(f"plan not found: {plan_path}")

    config_path = source_dir / CONFIG_NAME
    index_path = source_dir / INDEX_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"missing source config: {config_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"missing source index: {index_path}")

    retained_by_layer, keep_size, plan = parse_plan(
        plan_path,
        expected_layers=args.expected_layers,
        expected_original_experts=args.expected_original_experts,
    )

    source_config = load_json(config_path)
    text_config = get_text_config(source_config)
    source_num_experts = int(text_config.get("num_experts", -1))
    top_k = int(text_config.get("num_experts_per_tok", -1))

    if source_num_experts != args.expected_original_experts:
        raise ValueError(
            f"source config num_experts={source_num_experts}, expected "
            f"{args.expected_original_experts}"
        )
    if top_k <= 0:
        raise ValueError("could not read num_experts_per_tok from source config")
    if keep_size < top_k:
        raise ValueError(f"keep size {keep_size} is smaller than router top_k={top_k}")

    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model index is missing object field 'weight_map'")

    mtp_keys = sorted(key for key in weight_map if is_mtp_key(key))
    if mtp_keys:
        print(f"[info] detected MTP tensors: {len(mtp_keys)}")
        for key in mtp_keys[:20]:
            print(f"[info]   MTP: {key}")
        if len(mtp_keys) > 20:
            print(f"[info]   ... plus {len(mtp_keys) - 20} more")

    prunable_keys = discover_prunable_keys(
        weight_map,
        expected_layers=args.expected_layers,
    )

    shards = sorted(set(str(x) for x in weight_map.values()))
    missing_shards = [name for name in shards if not (source_dir / name).exists()]
    if missing_shards:
        raise FileNotFoundError(
            f"source snapshot is missing {len(missing_shards)} shard(s): {missing_shards[:5]}"
        )

    prepare_output_dir(output_dir, args.overwrite)
    copied_aux = copy_auxiliary_files(source_dir, output_dir)

    print(f"[info] source checkpoint: {source_dir}")
    print(f"[info] pruning plan: {plan_path}")
    print(f"[info] output checkpoint: {output_dir}")
    print(f"[info] routed experts per layer: {source_num_experts} -> {keep_size}")
    print(f"[info] router top_k remains: {top_k}")
    print(
        "[info] MTP policy: "
        + ("copied unchanged" if args.keep_mtp else "removed and disabled")
    )
    print(f"[info] shard count: {len(shards)}")

    modified_records: list[dict[str, Any]] = []
    removed_mtp_records: list[dict[str, Any]] = []
    output_weight_map: dict[str, str] = {}
    output_total_size = 0
    source_total_size_observed = 0

    for shard_index, shard_name in enumerate(shards, start=1):
        source_shard = source_dir / shard_name
        output_shard = output_dir / shard_name
        print(f"[shard {shard_index:02d}/{len(shards):02d}] {shard_name}")

        tensors: dict[str, torch.Tensor] = {}
        shard_metadata: dict[str, str] | None = None

        with safe_open(source_shard, framework="pt", device="cpu") as handle:
            shard_metadata = handle.metadata()
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                source_shape = list(tensor.shape)
                source_total_size_observed += tensor_nbytes(tensor)

                if is_mtp_key(key) and not args.keep_mtp:
                    removed_mtp_records.append(
                        {
                            "key": key,
                            "source_shape": source_shape,
                            "shard": shard_name,
                        }
                    )
                    continue

                if key in prunable_keys:
                    layer, kind = prunable_keys[key]
                    if tensor.ndim < 1 or int(tensor.shape[0]) != source_num_experts:
                        raise ValueError(
                            f"{key} shape {tuple(tensor.shape)} does not start with "
                            f"num_experts={source_num_experts}"
                        )
                    index_tensor = torch.tensor(
                        retained_by_layer[layer],
                        dtype=torch.long,
                    )
                    tensor = tensor.index_select(0, index_tensor).contiguous()
                    modified_records.append(
                        {
                            "key": key,
                            "layer": layer,
                            "kind": kind,
                            "source_shape": source_shape,
                            "output_shape": list(tensor.shape),
                            "shard": shard_name,
                        }
                    )
                else:
                    tensor = tensor.contiguous()

                output_total_size += tensor_nbytes(tensor)
                tensors[key] = tensor
                output_weight_map[key] = shard_name

        if tensors:
            save_file(tensors, output_shard, metadata=shard_metadata)
        else:
            print(f"[info] omitting empty output shard after MTP removal: {shard_name}")
        del tensors

    if len(modified_records) != args.expected_layers * 3:
        raise RuntimeError(
            f"modified {len(modified_records)} tensors; expected {args.expected_layers * 3}"
        )

    output_config = copy.deepcopy(source_config)
    output_text_config = get_text_config(output_config)
    output_text_config["num_experts"] = keep_size
    # Keep top_k unchanged explicitly.
    output_text_config["num_experts_per_tok"] = top_k
    if not args.keep_mtp:
        output_text_config["mtp_num_hidden_layers"] = 0
    if "num_experts" in output_config and output_config is not output_text_config:
        output_config["num_experts"] = keep_size
    write_json(output_dir / CONFIG_NAME, output_config)

    output_index = copy.deepcopy(index)
    output_index["weight_map"] = output_weight_map
    metadata = output_index.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        output_index["metadata"] = metadata
    metadata["total_size"] = output_total_size
    write_json(output_dir / INDEX_NAME, output_index)

    manifest = {
        "format_version": 1,
        "operation": "physical_routed_expert_pruning",
        "source_model_argument": args.source_model,
        "source_checkpoint_resolved": str(source_dir),
        "source_index_sha256": sha256_file(index_path),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "plan_criterion": plan.get("criterion"),
        "plan_bucket": plan.get("bucket"),
        "output_checkpoint": str(output_dir),
        "num_layers": args.expected_layers,
        "original_num_routed_experts_per_layer": source_num_experts,
        "retained_num_routed_experts_per_layer": keep_size,
        "num_experts_per_token_top_k": top_k,
        "shared_experts": "copied_unchanged",
        "mtp_policy": "copied_unchanged" if args.keep_mtp else "removed_and_disabled",
        "removed_mtp_tensor_count": len(removed_mtp_records),
        "removed_mtp_tensors": removed_mtp_records,
        "source_total_size_observed_bytes": source_total_size_observed,
        "output_total_size_bytes": output_total_size,
        "modified_tensor_count": len(modified_records),
        "modified_tensors": sorted(modified_records, key=lambda x: (x["layer"], x["kind"])),
        "copied_auxiliary_files": copied_aux,
        "retained_original_expert_ids_by_layer": {
            str(layer): ids for layer, ids in retained_by_layer.items()
        },
    }
    write_json(output_dir / "pruning_manifest.json", manifest)
    shutil.copy2(plan_path, output_dir / "pruning_plan.json")

    # Final lightweight verification directly from output shards.
    verified = 0
    for key, (layer, _kind) in prunable_keys.items():
        shard_name = weight_map[key]
        with safe_open(output_dir / shard_name, framework="pt", device="cpu") as handle:
            shape = handle.get_slice(key).get_shape()
        if int(shape[0]) != keep_size:
            raise RuntimeError(f"output verification failed for {key}: shape={shape}")
        verified += 1

    print("[done] checkpoint pruning completed")
    print(f"[done] verified pruned tensors: {verified}")
    print(f"[done] removed MTP tensors: {len(removed_mtp_records)}")
    print(f"[done] output size from tensor metadata: {output_total_size / 1e9:.3f} GB")
    print(f"[done] checkpoint: {output_dir}")
    print(f"[done] manifest: {output_dir / 'pruning_manifest.json'}")


if __name__ == "__main__":
    main()
