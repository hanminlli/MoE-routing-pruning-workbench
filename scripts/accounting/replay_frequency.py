from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import replay_count_experts_one_call as base  # type: ignore


BUCKET_PROMPT = "prompt_input"
BUCKET_GENERATED = "generated_output_prediction"


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().split('\n') if x.strip()]


def parse_indices(spec: str, available: list[int]) -> list[int]:
    if spec == "all":
        return available

    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))

    return sorted(dict.fromkeys(out))


def get_token_ids(call: dict[str, Any]) -> tuple[list[int], list[int]]:
    replay = call.get("replay", {}) if isinstance(call.get("replay"), dict) else {}

    prompt_ids = replay.get("prompt_token_ids_from_response")
    generated_ids = replay.get("generated_token_ids_from_response")

    if not isinstance(prompt_ids, list):
        response_token_ids = call.get("response_token_ids", {})
        if isinstance(response_token_ids, dict):
            prompt_ids = response_token_ids.get("prompt_token_ids")

    if not isinstance(generated_ids, list):
        response_token_ids = call.get("response_token_ids", {})
        if isinstance(response_token_ids, dict):
            generated_ids = response_token_ids.get("generated_token_ids")

    if not isinstance(prompt_ids, list):
        raise ValueError(f"call_index={call.get('call_index')} missing prompt token ids")

    if not isinstance(generated_ids, list):
        raise ValueError(f"call_index={call.get('call_index')} missing generated token ids")

    return [int(x) for x in prompt_ids], [int(x) for x in generated_ids]


class BucketCounter:
    def __init__(
        self,
        *,
        call_index: int,
        bucket_positions: dict[str, list[int]],
        top_k: int,
        expected_num_experts: int | None,
    ):
        self.call_index = call_index
        self.bucket_positions = bucket_positions
        self.top_k = int(top_k)
        self.expected_num_experts = expected_num_experts
        self.stats: dict[tuple[int, str, str], dict[str, Any]] = {}

    def ensure(
        self,
        *,
        bucket: str,
        module_name: str,
        layer_id: int,
        num_experts: int,
        tokens: int,
    ) -> dict[str, Any]:
        key = (self.call_index, bucket, module_name)

        if key not in self.stats:
            self.stats[key] = {
                "call_index": self.call_index,
                "bucket": bucket,
                "layer": layer_id,
                "module_name": module_name,
                "num_experts": num_experts,
                "tokens": int(tokens),
                "selected_count": [0 for _ in range(num_experts)],
                "weighted_count": [0.0 for _ in range(num_experts)],
                "rank_counts": [[0 for _ in range(num_experts)] for _ in range(self.top_k)],
            }

        return self.stats[key]

    def iter_tensors(self, obj: Any):
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from self.iter_tensors(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from self.iter_tensors(v)

    def looks_like_router_logits(self, x: torch.Tensor) -> bool:
        return (
            torch.is_floating_point(x)
            and x.ndim >= 2
            and self.expected_num_experts is not None
            and int(x.shape[-1]) == int(self.expected_num_experts)
        )

    def looks_like_expert_indices(self, x: torch.Tensor) -> bool:
        if self.expected_num_experts is None:
            return False
        if x.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.long}:
            return False
        if x.numel() == 0 or x.numel() > 10_000_000:
            return False

        with torch.no_grad():
            return int(x.min().item()) >= 0 and int(x.max().item()) < int(self.expected_num_experts)

    @torch.no_grad()
    def add_output(self, module_name: str, layer_id: int, output: Any) -> None:
        tensors = list(self.iter_tensors(output))

        for x in tensors:
            if self.looks_like_router_logits(x):
                self.add_logits(module_name, layer_id, x)
                return

        for x in tensors:
            if self.looks_like_expert_indices(x):
                self.add_indices(module_name, layer_id, x)
                return

    @torch.no_grad()
    def add_logits(self, module_name: str, layer_id: int, logits: torch.Tensor) -> None:
        flat = logits.detach().float().reshape(-1, logits.shape[-1])
        seq_len = int(flat.shape[0])
        num_experts = int(flat.shape[-1])
        k = min(self.top_k, num_experts)

        probs = torch.softmax(flat, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=k, dim=-1)
        denom = top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        top_weights = top_probs / denom

        for bucket, positions in self.bucket_positions.items():
            positions = [p for p in positions if 0 <= p < seq_len]
            if not positions:
                continue

            pos = torch.tensor(positions, dtype=torch.long, device=flat.device)
            idx = top_idx.index_select(0, pos)
            weights = top_weights.index_select(0, pos)

            stat = self.ensure(
                bucket=bucket,
                module_name=module_name,
                layer_id=layer_id,
                num_experts=num_experts,
                tokens=len(positions),
            )

            selected = idx.reshape(-1)
            counts = torch.bincount(selected, minlength=num_experts).cpu().tolist()

            weighted = torch.zeros(num_experts, dtype=torch.float32, device=flat.device)
            weighted.scatter_add_(0, selected, weights.reshape(-1))
            weighted_counts = weighted.cpu().tolist()

            for expert_id, value in enumerate(counts):
                stat["selected_count"][expert_id] += int(value)

            for expert_id, value in enumerate(weighted_counts):
                stat["weighted_count"][expert_id] += float(value)

            for rank in range(k):
                rank_counts = torch.bincount(idx[:, rank], minlength=num_experts).cpu().tolist()
                for expert_id, value in enumerate(rank_counts):
                    stat["rank_counts"][rank][expert_id] += int(value)

    @torch.no_grad()
    def add_indices(self, module_name: str, layer_id: int, indices: torch.Tensor) -> None:
        if self.expected_num_experts is None:
            return

        num_experts = int(self.expected_num_experts)
        idx = indices.detach().long()

        if idx.ndim >= 2 and idx.shape[-1] <= self.top_k:
            flat_2d = idx.reshape(-1, idx.shape[-1])
            k = int(flat_2d.shape[-1])
        else:
            flat_2d = idx.reshape(-1, 1)
            k = 1

        seq_len = int(flat_2d.shape[0])

        for bucket, positions in self.bucket_positions.items():
            positions = [p for p in positions if 0 <= p < seq_len]
            if not positions:
                continue

            pos = torch.tensor(positions, dtype=torch.long, device=flat_2d.device)
            idx_t = flat_2d.index_select(0, pos)

            stat = self.ensure(
                bucket=bucket,
                module_name=module_name,
                layer_id=layer_id,
                num_experts=num_experts,
                tokens=len(positions),
            )

            flat = idx_t.reshape(-1)
            valid = (flat >= 0) & (flat < num_experts)
            flat = flat[valid]

            counts = torch.bincount(flat, minlength=num_experts).cpu().tolist()

            for expert_id, value in enumerate(counts):
                stat["selected_count"][expert_id] += int(value)
                stat["weighted_count"][expert_id] += float(value) / max(k, 1)

            for rank in range(min(k, self.top_k)):
                rank_idx = idx_t[:, rank]
                valid_rank = (rank_idx >= 0) & (rank_idx < num_experts)
                rank_idx = rank_idx[valid_rank]
                rank_counts = torch.bincount(rank_idx, minlength=num_experts).cpu().tolist()

                for expert_id, value in enumerate(rank_counts):
                    stat["rank_counts"][rank][expert_id] += int(value)


def write_expert_csv(path: Path, stats: dict[tuple[int, str, str], dict[str, Any]], top_k: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = [
            "call_index",
            "bucket",
            "layer",
            "module_name",
            "expert",
            "tokens",
            "top_k",
            "selected_count",
            "activation_rate_per_token",
            "selection_share_within_bucket_layer",
            "weighted_count",
        ]
        header += [f"rank_{r}_count" for r in range(top_k)]
        writer.writerow(header)

        for _, stat in sorted(
            stats.items(),
            key=lambda kv: (kv[1]["call_index"], kv[1]["bucket"], kv[1]["layer"], kv[1]["module_name"]),
        ):
            tokens = int(stat["tokens"])
            denom = max(tokens * top_k, 1)

            for expert_id in range(int(stat["num_experts"])):
                selected_count = int(stat["selected_count"][expert_id])

                row = [
                    int(stat["call_index"]),
                    stat["bucket"],
                    int(stat["layer"]),
                    stat["module_name"],
                    expert_id,
                    tokens,
                    top_k,
                    selected_count,
                    selected_count / max(tokens, 1),
                    selected_count / denom,
                    float(stat["weighted_count"][expert_id]),
                ]

                for r in range(top_k):
                    row.append(int(stat["rank_counts"][r][expert_id]))

                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run_config.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-calls", required=True)
    parser.add_argument("--call-indices", default="all")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    cfg = base.load_json(Path(args.config)) if Path(args.config).exists() else {}
    model_name = args.model or cfg.get("model")
    if not model_name:
        raise ValueError("Model name missing. Provide --model or config['model'].")

    calls_all = iter_jsonl(Path(args.model_calls))
    by_index = {int(c["call_index"]): c for c in calls_all}
    wanted = parse_indices(args.call_indices, sorted(by_index))
    calls = [by_index[i] for i in wanted]

    print(f"[info] loading model once: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.eval()

    # Expert accounting only needs router modules to run.
    # Calling AutoModelForCausalLM.forward computes huge vocabulary logits
    # of shape [batch, seq_len, vocab_size], which is unnecessary and can OOM.
    # Use the base transformer body when available.
    forward_model = getattr(model, "model", None)
    if forward_model is None:
        forward_model = getattr(model, "transformer", None)
    if forward_model is None:
        forward_model = getattr(model, "base_model", None)
    if forward_model is None:
        forward_model = model

    if forward_model is model:
        print("[warn] using full CausalLM model for replay; LM-head logits may consume extra memory")
    else:
        print(f"[info] using base transformer for replay: {type(forward_model).__name__}")

    config = model.config

    num_experts = base.first_config_attr(
        config,
        ["num_experts", "n_routed_experts", "num_routed_experts", "moe_num_experts"],
    )
    num_experts = int(num_experts) if num_experts is not None else None

    top_k = args.top_k or base.first_config_attr(
        config,
        ["num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k", "router_top_k"],
    )
    if top_k is None:
        raise ValueError("Could not infer top_k; pass --top-k")
    top_k = int(top_k)

    routers = base.find_router_modules(model, expected_num_experts=num_experts)
    if not routers:
        raise RuntimeError("No MoE router modules found.")

    print(f"[info] num_experts={num_experts} top_k={top_k} routers={len(routers)}")

    first_device = next(model.parameters()).device

    current_counter: BucketCounter | None = None
    all_stats: dict[tuple[int, str, str], dict[str, Any]] = {}
    call_rows: list[dict[str, Any]] = []

    handles = []

    for module_name, layer_id, module in routers:
        def make_hook(name: str, lid: int):
            def hook(mod, inputs, output):
                if current_counter is not None:
                    current_counter.add_output(name, lid, output)
            return hook

        handles.append(module.register_forward_hook(make_hook(module_name, layer_id)))

    try:
        for call in calls:
            call_index = int(call["call_index"])
            prompt_ids, generated_ids = get_token_ids(call)

            n_prompt = len(prompt_ids)
            n_generated = len(generated_ids)

            if n_prompt <= 0:
                raise ValueError(f"call_index={call_index} has empty prompt ids")

            full_ids = prompt_ids + generated_ids

            # Meaning-2 accounting:
            # - Prompt/input bucket counts router positions 0 ... N-1.
            # - Generated/output bucket counts prediction positions N-1 ... N+T-2.
            #   This is length T and includes the final prompt token position for y1.
            prompt_positions = list(range(0, n_prompt))
            generated_positions = (
                list(range(n_prompt - 1, n_prompt + n_generated - 1))
                if n_generated > 0
                else []
            )

            bucket_positions = {
                BUCKET_PROMPT: prompt_positions,
                BUCKET_GENERATED: generated_positions,
            }

            usage = call.get("response", {}).get("token_usage", {})
            logged_input = int(usage.get("input", 0) or 0)
            logged_answer = int(usage.get("answer", 0) or 0)
            logged_reasoning = int(usage.get("reasoning", 0) or 0)
            logged_generated = logged_answer + logged_reasoning

            call_rows.append(
                {
                    "call_index": call_index,
                    "prompt_tokens": n_prompt,
                    "generated_tokens": n_generated,
                    "full_sequence_tokens": len(full_ids),
                    "logged_input_tokens": logged_input,
                    "logged_answer_tokens": logged_answer,
                    "logged_reasoning_tokens": logged_reasoning,
                    "logged_generated_tokens": logged_generated,
                    "prompt_delta": n_prompt - logged_input,
                    "generated_delta": n_generated - logged_generated,
                    "generated_prediction_positions": len(generated_positions),
                }
            )

            if n_prompt != logged_input:
                raise ValueError(
                    f"call_index={call_index} prompt id length {n_prompt} != logged input {logged_input}"
                )

            if n_generated != logged_generated:
                raise ValueError(
                    f"call_index={call_index} generated id length {n_generated} != logged generated {logged_generated}"
                )

            print(
                f"[info] call={call_index} prompt={n_prompt} generated={n_generated} "
                f"full={len(full_ids)}"
            )

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=first_device)
            attention_mask = torch.ones_like(input_ids, device=first_device)

            current_counter = BucketCounter(
                call_index=call_index,
                bucket_positions=bucket_positions,
                top_k=top_k,
                expected_num_experts=num_experts,
            )

            with torch.inference_mode():
                out = forward_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=False,
                )

            # Do not keep hidden states or output objects alive across calls.
            del out
            del input_ids
            del attention_mask

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            all_stats.update(current_counter.stats)
            current_counter = None

    finally:
        for h in handles:
            h.remove()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expert_long_path = out_dir / "expert_counts_prompt_and_generated_long.csv"
    write_expert_csv(expert_long_path, all_stats, top_k=top_k)

    import pandas as pd

    call_summary = pd.DataFrame(call_rows).sort_values("call_index")
    call_summary.to_csv(out_dir / "call_token_summary.csv", index=False)

    df = pd.read_csv(expert_long_path)

    rank_cols = [c for c in df.columns if c.startswith("rank_") and c.endswith("_count")]
    sum_cols = ["selected_count", "weighted_count"] + rank_cols

    agg = df.groupby(
        ["bucket", "layer", "module_name", "expert"],
        as_index=False,
    ).agg({
        **{c: "sum" for c in sum_cols},
        "tokens": "sum",
        "top_k": "first",
    })

    agg["activation_rate_per_token"] = agg["selected_count"] / agg["tokens"].clip(lower=1)
    agg["selection_share_within_bucket_layer"] = agg["selected_count"] / (
        agg["tokens"].clip(lower=1) * agg["top_k"]
    )

    agg = agg.sort_values(
        ["bucket", "layer", "selected_count"],
        ascending=[True, True, False],
    )

    agg.to_csv(out_dir / "expert_counts_prompt_and_generated_aggregated.csv", index=False)

    bucket_summary = (
        call_summary[["prompt_tokens", "generated_tokens"]]
        .sum()
        .to_dict()
    )

    num_layers = len(routers)

    prompt_total_selected = int(agg[agg["bucket"] == BUCKET_PROMPT]["selected_count"].sum())
    generated_total_selected = int(agg[agg["bucket"] == BUCKET_GENERATED]["selected_count"].sum())

    expected_prompt = int(bucket_summary["prompt_tokens"] * top_k * num_layers)
    expected_generated = int(bucket_summary["generated_tokens"] * top_k * num_layers)

    metadata = {
        "mode": "prompt_input_and_generated_output_prediction_expert_counting",
        "meaning": {
            "prompt_input": "router activations on prompt positions 0..N-1",
            "generated_output_prediction": "router activations on prediction positions N-1..N+T-2, one position per generated token",
            "note": "prompt_input and generated_output_prediction overlap at the last prompt token for each call because that position predicts the first generated token.",
        },
        "model": model_name,
        "model_calls_path": str(args.model_calls),
        "call_indices": wanted,
        "num_calls": len(wanted),
        "num_experts": num_experts,
        "top_k": top_k,
        "num_router_modules": num_layers,
        "total_prompt_tokens": int(bucket_summary["prompt_tokens"]),
        "total_generated_tokens": int(bucket_summary["generated_tokens"]),
        "prompt_total_selected_count": prompt_total_selected,
        "expected_prompt_total_selected_count": expected_prompt,
        "prompt_matches_expected_total": prompt_total_selected == expected_prompt,
        "generated_total_selected_count": generated_total_selected,
        "expected_generated_total_selected_count": expected_generated,
        "generated_matches_expected_total": generated_total_selected == expected_generated,
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[done] wrote:")
    for name in [
        "metadata.json",
        "call_token_summary.csv",
        "expert_counts_prompt_and_generated_long.csv",
        "expert_counts_prompt_and_generated_aggregated.csv",
    ]:
        print(" ", out_dir / name)

    print()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
