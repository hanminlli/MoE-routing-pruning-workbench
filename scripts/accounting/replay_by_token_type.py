from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import replay_count_experts_one_call as base  # type: ignore


TOKEN_TYPES = [
    "control",
    "system_prompt",
    "tool_schema",
    "user_prompt",
    "assistant_content",
    "assistant_reasoning",
    "assistant_tool_call",
    "tool_result",
    "other",
]


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    calls = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                calls.append(json.loads(line))
    return calls


def parse_indices(spec: str, available: list[int]) -> list[int]:
    if spec == "all":
        return available
    out = []
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


def mark(labels: list[str], start: int, end: int, label: str) -> None:
    start = max(0, start)
    end = min(len(labels), end)
    for i in range(start, end):
        labels[i] = label


def classify_rendered_prompt_chars(rendered: str) -> list[str]:
    labels = ["other"] * len(rendered)

    role_pat = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)(?=<\|im_end\|>)", re.S)

    for m in role_pat.finditer(rendered):
        role = m.group(1)
        body_start, body_end = m.start(2), m.end(2)

        if role == "system":
            label = "system_prompt"
        elif role == "user":
            label = "user_prompt"
        elif role == "assistant":
            label = "assistant_content"
        elif role == "tool":
            label = "tool_result"
        else:
            label = "other"

        mark(labels, body_start, body_end, label)
        mark(labels, m.start(0), body_start, "control")

        if role == "system":
            body = rendered[body_start:body_end]
            tool_anchor = body.find("# Tools")
            if tool_anchor == -1:
                tool_anchor = body.find("You may call one or more functions")
            if tool_anchor != -1:
                mark(labels, body_start + tool_anchor, body_end, "tool_schema")

    for tag, label in [
        ("think", "assistant_reasoning"),
        ("tool_call", "assistant_tool_call"),
        ("tool_response", "tool_result"),
    ]:
        pat = re.compile(rf"<{tag}>\n?(.*?)\n?</{tag}>", re.S)
        for m in pat.finditer(rendered):
            mark(labels, m.start(1), m.end(1), label)

    control_patterns = [
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"<think>",
        r"</think>",
        r"<tool_call>",
        r"</tool_call>",
        r"<tool_response>",
        r"</tool_response>",
    ]
    for pat in control_patterns:
        for m in re.finditer(pat, rendered):
            mark(labels, m.start(), m.end(), "control")

    return labels


def token_types_from_offsets(rendered: str, offsets: list[tuple[int, int]]) -> list[str]:
    char_labels = classify_rendered_prompt_chars(rendered)
    token_types = []

    for s, e in offsets:
        s, e = int(s), int(e)
        if e <= s or s < 0 or e > len(char_labels):
            token_types.append("control")
            continue

        vals = char_labels[s:e]
        token_types.append(Counter(vals).most_common(1)[0][0] if vals else "other")

    return token_types


def extract_reasoning_texts(obj: Any) -> list[str]:
    texts = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if lk in {"reasoning_content", "reasoning", "thought", "thinking"} and isinstance(v, str):
                    if v.strip():
                        texts.append(v.strip())
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            for m in re.finditer(r"<think>\n?(.*?)\n?</think>", x, flags=re.S):
                t = m.group(1).strip()
                if t:
                    texts.append(t)

    walk(obj)

    seen = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


class TypedExpertCounter:
    def __init__(self, token_types: list[str], top_k: int, expected_num_experts: int | None):
        self.token_types = token_types
        self.top_k = int(top_k)
        self.expected_num_experts = expected_num_experts
        self.stats: dict[tuple[int, str, str], dict[str, Any]] = {}

    def ensure(self, call_index: int, token_type: str, module_name: str, layer_id: int, num_experts: int, tokens: int):
        key = (call_index, token_type, module_name)
        if key not in self.stats:
            self.stats[key] = {
                "call_index": call_index,
                "token_type": token_type,
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
    def add_output(self, call_index: int, module_name: str, layer_id: int, output: Any) -> None:
        tensors = list(self.iter_tensors(output))

        for x in tensors:
            if self.looks_like_router_logits(x):
                self.add_logits(call_index, module_name, layer_id, x)
                return

        for x in tensors:
            if self.looks_like_expert_indices(x):
                self.add_indices(call_index, module_name, layer_id, x)
                return

    @torch.no_grad()
    def add_logits(self, call_index: int, module_name: str, layer_id: int, logits: torch.Tensor) -> None:
        flat = logits.detach().float().reshape(-1, logits.shape[-1])
        num_tokens = int(flat.shape[0])
        num_experts = int(flat.shape[-1])

        if num_tokens != len(self.token_types):
            raise RuntimeError(
                f"router token count mismatch for {module_name}: router={num_tokens}, prompt={len(self.token_types)}"
            )

        k = min(self.top_k, num_experts)
        probs = torch.softmax(flat, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=k, dim=-1)

        denom = top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        top_weights = top_probs / denom

        for token_type in sorted(set(self.token_types)):
            positions = [i for i, t in enumerate(self.token_types) if t == token_type]
            n = len(positions)
            if n == 0:
                continue

            pos = torch.tensor(positions, dtype=torch.long, device=flat.device)
            idx = top_idx.index_select(0, pos)
            weights = top_weights.index_select(0, pos)

            stat = self.ensure(call_index, token_type, module_name, layer_id, num_experts, n)

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
    def add_indices(self, call_index: int, module_name: str, layer_id: int, indices: torch.Tensor) -> None:
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

        if int(flat_2d.shape[0]) != len(self.token_types):
            raise RuntimeError(
                f"router token count mismatch for {module_name}: router={int(flat_2d.shape[0])}, prompt={len(self.token_types)}"
            )

        for token_type in sorted(set(self.token_types)):
            positions = [i for i, t in enumerate(self.token_types) if t == token_type]
            n = len(positions)
            if n == 0:
                continue

            pos = torch.tensor(positions, dtype=torch.long, device=flat_2d.device)
            idx_t = flat_2d.index_select(0, pos)

            stat = self.ensure(call_index, token_type, module_name, layer_id, num_experts, n)

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
            "token_type",
            "layer",
            "module_name",
            "expert",
            "tokens",
            "top_k",
            "selected_count",
            "activation_rate_per_token",
            "selection_share_within_type_layer",
            "weighted_count",
        ]
        header += [f"rank_{r}_count" for r in range(top_k)]
        writer.writerow(header)

        for _, stat in sorted(
            stats.items(),
            key=lambda kv: (kv[1]["call_index"], kv[1]["token_type"], kv[1]["layer"], kv[1]["module_name"]),
        ):
            tokens = int(stat["tokens"])
            denom = max(tokens * top_k, 1)

            for expert_id in range(int(stat["num_experts"])):
                selected_count = int(stat["selected_count"][expert_id])

                row = [
                    int(stat["call_index"]),
                    stat["token_type"],
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
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--schema-mode", default="known")
    parser.add_argument("--enable-thinking", choices=["true", "false", "none"], default="none")
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = base.load_json(Path(args.config)) if Path(args.config).exists() else {}
    model_name = args.model or cfg.get("model")
    if not model_name:
        raise ValueError("Model name missing. Provide --model or config['model'].")

    chat_template_path = args.chat_template or cfg.get("chat_template_path")
    enable_thinking = None if args.enable_thinking == "none" else (args.enable_thinking == "true")

    calls_all = iter_jsonl(Path(args.model_calls))
    by_index = {int(c["call_index"]): c for c in calls_all}
    wanted = parse_indices(args.call_indices, sorted(by_index))
    calls = [by_index[i] for i in wanted]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    if chat_template_path:
        tokenizer.chat_template = Path(chat_template_path).read_text(encoding="utf-8")
        print(f"[info] using chat template: {chat_template_path}")

    print(f"[info] loading model once: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.eval()

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

    all_stats = {}
    token_type_rows = []
    reasoning_rows = []

    current_counter = None
    current_call_index = None

    handles = []

    for module_name, layer_id, module in routers:
        def make_hook(name: str, lid: int):
            def hook(mod, inputs, output):
                if current_counter is None or current_call_index is None:
                    return
                current_counter.add_output(current_call_index, name, lid, output)
            return hook

        handles.append(module.register_forward_hook(make_hook(module_name, layer_id)))

    try:
        for call in calls:
            call_index = int(call["call_index"])
            request = call["request"]

            messages = base.sanitize_messages(request.get("messages", []))
            tools = base.normalize_logged_tools(request.get("tools"), schema_mode=args.schema_mode)

            replay = call.get("replay") if isinstance(call.get("replay"), dict) else {}
            rendered = replay.get("rendered_prompt")
            used_logged_rendered = isinstance(rendered, str) and bool(rendered)

            if not used_logged_rendered:
                rendered, _ = base.render_prompt(
                    tokenizer=tokenizer,
                    messages=messages,
                    tools=tools,
                    no_tools=args.no_tools,
                    enable_thinking=enable_thinking,
                )

            enc_with_offsets = tokenizer(
                rendered,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
            )

            offsets = [
                (int(a), int(b))
                for a, b in enc_with_offsets.pop("offset_mapping")[0].tolist()
            ]

            token_types = token_types_from_offsets(rendered, offsets)
            token_counts = Counter(token_types)

            input_tokens = int(enc_with_offsets["input_ids"].shape[1])

            logged_input_tokens = None
            try:
                logged_input_tokens = int(call["response"]["token_usage"]["input"])
            except Exception:
                pass

            for token_type in TOKEN_TYPES:
                token_type_rows.append({
                    "call_index": call_index,
                    "token_type": token_type,
                    "tokens": int(token_counts.get(token_type, 0)),
                    "input_tokens": input_tokens,
                    "logged_input_tokens": logged_input_tokens,
                    "token_delta": None if logged_input_tokens is None else input_tokens - logged_input_tokens,
                    "used_logged_rendered_prompt": used_logged_rendered,
                })

            reasoning_texts = extract_reasoning_texts(call.get("response", {}))
            reasoning_tokens = sum(
                len(tokenizer(t, add_special_tokens=False)["input_ids"])
                for t in reasoning_texts
            )

            output_tokens = None
            try:
                output_tokens = int(call["response"]["token_usage"].get("output"))
            except Exception:
                pass

            reasoning_rows.append({
                "call_index": call_index,
                "generated_reasoning_text_blocks_found": len(reasoning_texts),
                "generated_reasoning_tokens_from_logged_text": int(reasoning_tokens),
                "logged_output_tokens": output_tokens,
            })

            print(
                f"[info] call={call_index} input_tokens={input_tokens} "
                f"logged={logged_input_tokens} "
                f"delta={None if logged_input_tokens is None else input_tokens - logged_input_tokens} "
                f"generated_reasoning_tokens={reasoning_tokens}"
            )

            current_call_index = call_index
            current_counter = TypedExpertCounter(
                token_types=token_types,
                top_k=top_k,
                expected_num_experts=num_experts,
            )

            enc = {k: v.to(first_device) for k, v in enc_with_offsets.items()}

            with torch.no_grad():
                _ = model(**enc, use_cache=False)

            all_stats.update(current_counter.stats)
            current_counter = None
            current_call_index = None

    finally:
        for h in handles:
            h.remove()

    with (out_dir / "prefill_token_type_counts_by_call.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(token_type_rows[0].keys()))
        writer.writeheader()
        writer.writerows(token_type_rows)

    with (out_dir / "generated_reasoning_token_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(reasoning_rows[0].keys()))
        writer.writeheader()
        writer.writerows(reasoning_rows)

    expert_long = out_dir / "expert_counts_by_token_type_long.csv"
    write_expert_csv(expert_long, all_stats, top_k=top_k)

    import pandas as pd

    df = pd.read_csv(expert_long)

    rank_cols = [c for c in df.columns if c.startswith("rank_") and c.endswith("_count")]
    agg_cols = ["selected_count", "weighted_count"] + rank_cols

    agg = df.groupby(
        ["token_type", "layer", "module_name", "expert"],
        as_index=False,
    ).agg({
        **{c: "sum" for c in agg_cols},
        "tokens": "sum",
        "top_k": "first",
    })

    agg["activation_rate_per_token"] = agg["selected_count"] / agg["tokens"].clip(lower=1)
    agg["selection_share_within_type_layer"] = agg["selected_count"] / (
        agg["tokens"].clip(lower=1) * agg["top_k"]
    )

    agg = agg.sort_values(
        ["token_type", "layer", "selected_count"],
        ascending=[True, True, False],
    )

    agg.to_csv(out_dir / "expert_counts_by_token_type_aggregated.csv", index=False)

    type_counts = pd.read_csv(out_dir / "prefill_token_type_counts_by_call.csv")
    type_summary = (
        type_counts.groupby("token_type", as_index=False)["tokens"]
        .sum()
        .sort_values("tokens", ascending=False)
    )
    type_summary.to_csv(out_dir / "prefill_token_type_summary.csv", index=False)

    reasoning_df = pd.read_csv(out_dir / "generated_reasoning_token_summary.csv")

    metadata = {
        "mode": "prefill_expert_count_by_token_type",
        "model": model_name,
        "model_calls_path": str(args.model_calls),
        "call_indices": wanted,
        "num_calls": len(wanted),
        "num_experts": num_experts,
        "top_k": top_k,
        "num_router_modules": len(routers),
        "token_types": TOKEN_TYPES,
        "prefill_token_type_totals": dict(
            zip(type_summary["token_type"], [int(x) for x in type_summary["tokens"]])
        ),
        "generated_reasoning_tokens_from_logged_text_total": int(
            reasoning_df["generated_reasoning_tokens_from_logged_text"].sum()
        ),
        "note": (
            "Expert counts are for prefill input tokens grouped by rendered-prompt token type. "
            "Generated reasoning token counts are reported only if reasoning text was present in the logs. "
            "Decode-token expert routing is not included here."
        ),
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[done] wrote:")
    for name in [
        "metadata.json",
        "prefill_token_type_counts_by_call.csv",
        "prefill_token_type_summary.csv",
        "generated_reasoning_token_summary.csv",
        "expert_counts_by_token_type_long.csv",
        "expert_counts_by_token_type_aggregated.csv",
    ]:
        print(" ", out_dir / name)


if __name__ == "__main__":
    main()
