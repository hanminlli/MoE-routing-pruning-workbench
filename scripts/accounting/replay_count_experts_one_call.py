from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_call(path: Path, call_index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if int(obj.get("call_index", -1)) == call_index:
                return obj

    raise ValueError(f"call_index={call_index} not found in {path}")


def sanitize_messages(messages: Any) -> list[dict[str, Any]]:
    """
    Convert logged Stirrup/vLLM messages into the shape expected by the
    Qwen HF chat template.

    Important for calls after the first one:
    logged assistant tool_calls often have arguments as a JSON string, e.g.
        {"name": "code_exec", "arguments": "{\"cmd\": \"...\"}"}
    but Qwen's Jinja template expects arguments to be a mapping/dict.
    """
    import json

    def parse_arguments(x: Any) -> Any:
        if isinstance(x, str):
            try:
                parsed = json.loads(x)
                if isinstance(parsed, dict):
                    return parsed
                return {"value": parsed}
            except Exception:
                return {"_raw": x}
        if isinstance(x, dict):
            return x
        if x is None:
            return {}
        return {"value": x}

    def normalize_tool_call(tc: Any) -> dict[str, Any]:
        if not isinstance(tc, dict):
            return {"name": "unknown", "arguments": {"_raw": str(tc)}}

        out: dict[str, Any] = {}

        # Preserve identifiers if present.
        if tc.get("id") is not None:
            out["id"] = tc.get("id")
        elif tc.get("tool_call_id") is not None:
            out["id"] = tc.get("tool_call_id")

        if tc.get("type") is not None:
            out["type"] = tc.get("type")

        # Handle OpenAI-style nested function tool calls.
        fn = tc.get("function")
        if isinstance(fn, dict):
            name = fn.get("name") or tc.get("name") or "unknown"
            args = parse_arguments(fn.get("arguments", tc.get("arguments", {})))
            out["function"] = {"name": name, "arguments": args}
            # Also keep flat form for Qwen templates that expect flat calls.
            out["name"] = name
            out["arguments"] = args
            return out

        # Handle Stirrup/vLLM flat tool calls.
        name = tc.get("name") or "unknown"
        args = parse_arguments(tc.get("arguments", {}))
        out["name"] = name
        out["arguments"] = args
        return out

    out: list[dict[str, Any]] = []

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        m: dict[str, Any] = {"role": role}

        if msg.get("content") is not None:
            m["content"] = msg.get("content")
        elif role == "assistant":
            m["content"] = ""
        elif role == "tool":
            m["content"] = ""

        if role == "tool":
            if msg.get("tool_call_id") is not None:
                m["tool_call_id"] = msg.get("tool_call_id")
            if msg.get("name") is not None:
                m["name"] = msg.get("name")

        if role == "assistant" and msg.get("tool_calls"):
            m["tool_calls"] = [normalize_tool_call(tc) for tc in msg.get("tool_calls", [])]

        out.append(m)

    return out


def minimal_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


def drop_titles(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: drop_titles(v)
            for k, v in obj.items()
            if k != "title"
        }

    if isinstance(obj, list):
        return [drop_titles(v) for v in obj]

    return obj


def known_schema(name: str, mode: str, raw_parameters: Any = None) -> dict[str, Any]:
    """
    Reconstruct OpenAI/HF-style JSON schemas for the current Stirrup tools.

    Current logs store tool["parameters"] as a stringified Python mappingproxy,
    not a JSON schema. For already-recorded runs, reconstruct known schemas.
    """
    if mode == "minimal":
        return minimal_schema()

    if name == "finish":
        schema = {
            "type": "object",
            "title": "FinishParams",
            "properties": {
                "reason": {
                    "type": "string",
                    "title": "Reason",
                    "description": "Reason for finishing.",
                },
                "paths": {
                    "type": "array",
                    "title": "Paths",
                    "description": "List of file paths created or modified. Do not include directories, only files.",
                    "items": {"type": "string"},
                },
            },
            "required": ["reason", "paths"],
        }

    elif name == "code_exec":
        cmd_description = (
            "Shell command to execute (bash syntax). IMPORTANT: Use only relative paths. "
            "Do not use absolute paths (starting with / or ~) or reference directories "
            "outside the working directory."
        )

        if mode == "known_long_desc":
            cmd_description = (
                "Shell command to execute (bash syntax). IMPORTANT: Use only relative paths. "
                "Do not use absolute paths (starting with / or ~) or reference directories "
                "outside the working directory. Returns exit code, stdout, and stderr as XML."
            )

        schema = {
            "type": "object",
            "title": "CodeExecParams",
            "properties": {
                "cmd": {
                    "type": "string",
                    "title": "Cmd",
                    "description": cmd_description,
                },
            },
            "required": ["cmd"],
        }

    elif name == "fetch_web_page":
        url_description = "Full HTTP or HTTPS URL of the web page to fetch and extract."

        if mode == "known_long_desc":
            url_description = (
                "Full HTTP or HTTPS URL of the web page to fetch and extract. "
                "The tool fetches the web page and extracts the main content as markdown."
            )

        schema = {
            "type": "object",
            "title": "FetchWebPageParams",
            "properties": {
                "url": {
                    "type": "string",
                    "title": "Url",
                    "description": url_description,
                },
            },
            "required": ["url"],
        }

    else:
        schema = minimal_schema()

    if mode == "known_no_titles":
        schema = drop_titles(schema)

    if mode == "raw_params_description" and isinstance(raw_parameters, str):
        schema = {
            "type": "object",
            "title": f"{name}Parameters",
            "description": raw_parameters,
            "properties": {},
            "additionalProperties": True,
        }

    return schema


def normalize_logged_tools(logged_tools: Any, schema_mode: str = "known") -> list[dict[str, Any]]:
    """
    Convert logged Stirrup tools into the format expected by Qwen's HF chat template.
    """
    if not logged_tools:
        return []

    if isinstance(logged_tools, list):
        return logged_tools

    if not isinstance(logged_tools, dict):
        return []

    tools: list[dict[str, Any]] = []

    for fallback_name, spec in logged_tools.items():
        if isinstance(spec, dict):
            name = str(spec.get("name") or fallback_name)
            description = str(spec.get("description") or "")
            raw_parameters = spec.get("parameters")

            if isinstance(raw_parameters, dict):
                parameters = raw_parameters
            else:
                parameters = known_schema(name, schema_mode, raw_parameters=raw_parameters)
        else:
            name = str(fallback_name)
            description = ""
            raw_parameters = None
            parameters = known_schema(name, schema_mode, raw_parameters=raw_parameters)

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    return tools


def render_prompt(
    *,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    no_tools: bool,
    enable_thinking: bool | None,
) -> tuple[str, bool]:
    kwargs: dict[str, Any] = {}

    if tools and not no_tools:
        kwargs["tools"] = tools

    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )

    return rendered, bool(tools and not no_tools)


def first_config_attr(config: Any, names: list[str]) -> Any:
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value

    return None


def parse_layer_id(module_name: str, fallback: int) -> int:
    m = re.search(r"layers\.(\d+)", module_name)
    if m:
        return int(m.group(1))

    return fallback


def find_router_modules(model: torch.nn.Module, expected_num_experts: int | None) -> list[tuple[str, int, torch.nn.Module]]:
    """
    Find Qwen3.5/Qwen3.6 MoE router modules.

    Inspection shows the routed expert routers are:
        model.layers.<L>.mlp.gate | Qwen3_5MoeTopKRouter

    We intentionally exclude:
        shared_expert_gate
        shared_expert.gate_proj
        experts.*
    """
    routers: list[tuple[str, int, torch.nn.Module]] = []

    for name, module in model.named_modules():
        cls_name = type(module).__name__

        if not name.endswith(".mlp.gate"):
            continue

        if "TopKRouter" not in cls_name:
            continue

        layer_id = parse_layer_id(name, len(routers))
        routers.append((name, layer_id, module))

    return routers


class ExpertCounter:
    def __init__(self, top_k: int, expected_num_experts: int | None = None):
        self.top_k = int(top_k)
        self.expected_num_experts = expected_num_experts
        self.stats: dict[str, dict[str, Any]] = {}

    def ensure_module(self, module_name: str, layer_id: int, num_experts: int) -> None:
        if module_name in self.stats:
            return

        self.stats[module_name] = {
            "layer": layer_id,
            "module_name": module_name,
            "num_experts": num_experts,
            "tokens": 0,
            "selected_count": [0 for _ in range(num_experts)],
            "weighted_count": [0.0 for _ in range(num_experts)],
            "rank_counts": [
                [0 for _ in range(num_experts)]
                for _ in range(self.top_k)
            ],
        }

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
        if not torch.is_floating_point(x):
            return False

        if x.ndim < 2:
            return False

        if self.expected_num_experts is None:
            return False

        return int(x.shape[-1]) == int(self.expected_num_experts)

    def looks_like_expert_indices(self, x: torch.Tensor) -> bool:
        if x.ndim < 1:
            return False

        if self.expected_num_experts is None:
            return False

        if x.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.long}:
            return False

        if x.numel() == 0 or x.numel() > 10_000_000:
            return False

        with torch.no_grad():
            xmin = int(x.min().item())
            xmax = int(x.max().item())

        return xmin >= 0 and xmax < int(self.expected_num_experts)

    @torch.no_grad()
    def add_output(self, module_name: str, layer_id: int, output: Any) -> None:
        tensors = list(self.iter_tensors(output))

        # Preferred case: router returns logits of shape [..., num_experts].
        for x in tensors:
            if self.looks_like_router_logits(x):
                self.add_logits(module_name, layer_id, x)
                return

        # Fallback case: router returns selected expert ids directly.
        for x in tensors:
            if self.looks_like_expert_indices(x):
                self.add_indices(module_name, layer_id, x)
                return

    @torch.no_grad()
    def add_logits(self, module_name: str, layer_id: int, logits: torch.Tensor) -> None:
        flat = logits.detach().float().reshape(-1, logits.shape[-1])
        num_tokens = int(flat.shape[0])
        num_experts = int(flat.shape[-1])

        if self.expected_num_experts is not None and num_experts != int(self.expected_num_experts):
            return

        k = min(self.top_k, num_experts)

        self.ensure_module(module_name, layer_id, num_experts)
        stat = self.stats[module_name]
        stat["tokens"] += num_tokens

        probs = torch.softmax(flat, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=k, dim=-1)

        # Typical MoE implementation renormalizes selected top-k probabilities.
        denom = top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        top_weights = top_probs / denom

        selected = top_idx.reshape(-1)
        selected_count = torch.bincount(selected, minlength=num_experts).cpu().tolist()

        weighted = torch.zeros(num_experts, dtype=torch.float32, device=flat.device)
        weighted.scatter_add_(0, selected, top_weights.reshape(-1))
        weighted_count = weighted.cpu().tolist()

        for expert_id, value in enumerate(selected_count):
            stat["selected_count"][expert_id] += int(value)

        for expert_id, value in enumerate(weighted_count):
            stat["weighted_count"][expert_id] += float(value)

        for rank in range(k):
            rank_count = torch.bincount(top_idx[:, rank], minlength=num_experts).cpu().tolist()
            for expert_id, value in enumerate(rank_count):
                stat["rank_counts"][rank][expert_id] += int(value)

    @torch.no_grad()
    def add_indices(self, module_name: str, layer_id: int, indices: torch.Tensor) -> None:
        if self.expected_num_experts is None:
            return

        num_experts = int(self.expected_num_experts)
        idx = indices.detach().long()

        if idx.ndim >= 2 and idx.shape[-1] <= self.top_k:
            flat_2d = idx.reshape(-1, idx.shape[-1])
            num_tokens = int(flat_2d.shape[0])
            k = int(flat_2d.shape[-1])
        else:
            flat_2d = idx.reshape(-1, 1)
            num_tokens = int(flat_2d.shape[0])
            k = 1

        self.ensure_module(module_name, layer_id, num_experts)
        stat = self.stats[module_name]
        stat["tokens"] += num_tokens

        flat = flat_2d.reshape(-1)
        valid = (flat >= 0) & (flat < num_experts)
        flat = flat[valid]

        selected_count = torch.bincount(flat, minlength=num_experts).cpu().tolist()

        for expert_id, value in enumerate(selected_count):
            stat["selected_count"][expert_id] += int(value)
            stat["weighted_count"][expert_id] += float(value) / max(k, 1)

        for rank in range(min(k, self.top_k)):
            rank_idx = flat_2d[:, rank]
            valid_rank = (rank_idx >= 0) & (rank_idx < num_experts)
            rank_idx = rank_idx[valid_rank]
            rank_count = torch.bincount(rank_idx, minlength=num_experts).cpu().tolist()

            for expert_id, value in enumerate(rank_count):
                stat["rank_counts"][rank][expert_id] += int(value)


def write_outputs(
    *,
    out_dir: Path,
    metadata: dict[str, Any],
    counter: ExpertCounter,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (out_dir / "expert_counts_by_module.json").write_text(
        json.dumps(counter.stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = out_dir / "expert_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = [
            "layer",
            "module_name",
            "expert",
            "tokens",
            "top_k",
            "selected_count",
            "activation_rate_per_token",
            "selection_share_within_layer",
            "weighted_count",
        ]
        header += [f"rank_{r}_count" for r in range(counter.top_k)]
        writer.writerow(header)

        for module_name, stat in sorted(counter.stats.items(), key=lambda kv: (kv[1]["layer"], kv[0])):
            layer = int(stat["layer"])
            num_experts = int(stat["num_experts"])
            tokens = int(stat["tokens"])
            denom_selected = max(tokens * counter.top_k, 1)

            for expert_id in range(num_experts):
                selected_count = int(stat["selected_count"][expert_id])
                weighted_count = float(stat["weighted_count"][expert_id])

                row = [
                    layer,
                    module_name,
                    expert_id,
                    tokens,
                    counter.top_k,
                    selected_count,
                    selected_count / max(tokens, 1),
                    selected_count / denom_selected,
                    weighted_count,
                ]

                for r in range(counter.top_k):
                    row.append(int(stat["rank_counts"][r][expert_id]))

                writer.writerow(row)

    print(f"[ok] wrote {out_dir / 'metadata.json'}")
    print(f"[ok] wrote {out_dir / 'expert_counts_by_module.json'}")
    print(f"[ok] wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run_config.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-calls", required=True)
    parser.add_argument("--call-index", type=int, default=1)
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--dry-run-render-only", action="store_true", help="Only render/tokenize prompt and exit before loading model.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--schema-mode", default="known", choices=["minimal", "known", "known_no_titles", "known_long_desc", "raw_params_description"])
    parser.add_argument("--enable-thinking", choices=["true", "false", "none"], default="none")
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--ignore-logged-rendered-prompt", action="store_true")
    args = parser.parse_args()

    cfg = load_json(Path(args.config)) if Path(args.config).exists() else {}

    model_name = args.model or cfg.get("model")
    if not model_name:
        raise ValueError("Model name missing. Provide --model or config['model'].")

    chat_template_path = args.chat_template or cfg.get("chat_template_path")

    enable_thinking: bool | None
    if args.enable_thinking == "true":
        enable_thinking = True
    elif args.enable_thinking == "false":
        enable_thinking = False
    else:
        enable_thinking = None

    call = read_jsonl_call(Path(args.model_calls), args.call_index)
    request = call["request"]

    messages = sanitize_messages(request.get("messages", []))
    tools = normalize_logged_tools(request.get("tools"), schema_mode=args.schema_mode)

    print(f"[info] loaded call_index={args.call_index}")
    print(f"[info] messages={len(messages)}")
    print(f"[info] tools={len(tools)}")
    print(f"[info] schema_mode={args.schema_mode}")
    print(f"[info] enable_thinking={enable_thinking}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    if chat_template_path:
        template_text = Path(chat_template_path).read_text(encoding="utf-8")
        tokenizer.chat_template = template_text
        print(f"[info] using chat template: {chat_template_path}")

    replay_obj = call.get("replay") if isinstance(call.get("replay"), dict) else {}
    logged_rendered = replay_obj.get("rendered_prompt")

    if logged_rendered and not args.ignore_logged_rendered_prompt:
        rendered = str(logged_rendered)
        used_tools = bool(replay_obj.get("rendered_with_tools"))
        print("[info] using logged replay.rendered_prompt")
    else:
        rendered, used_tools = render_prompt(
            tokenizer=tokenizer,
            messages=messages,
            tools=tools,
            no_tools=args.no_tools,
            enable_thinking=enable_thinking,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rendered_prompt.txt").write_text(rendered, encoding="utf-8")
    (out_dir / "normalized_tools.json").write_text(
        json.dumps(tools, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    enc = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_tokens = int(enc["input_ids"].shape[1])

    logged_input_tokens = None
    try:
        logged_input_tokens = int(call["response"]["token_usage"]["input"])
    except Exception:
        pass

    print(f"[info] rendered prompt tokens: {input_tokens}")
    print(f"[info] logged input tokens: {logged_input_tokens}")
    if logged_input_tokens is not None:
        print(f"[info] token delta rendered-minus-logged: {input_tokens - logged_input_tokens}")
    print(f"[info] rendered with tools: {used_tools}")

    if args.dry_run_render_only:
        print("[done] dry run render/token check complete.")
        return

    print("[info] loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.eval()

    config = model.config

    num_experts = first_config_attr(
        config,
        [
            "num_experts",
            "n_routed_experts",
            "num_routed_experts",
            "moe_num_experts",
        ],
    )
    if num_experts is not None:
        num_experts = int(num_experts)

    top_k = args.top_k
    if top_k is None:
        top_k = first_config_attr(
            config,
            [
                "num_experts_per_tok",
                "num_experts_per_token",
                "moe_top_k",
                "top_k",
                "router_top_k",
            ],
        )

    if top_k is None:
        raise ValueError("Could not infer MoE top_k from model.config. Pass --top-k explicitly.")

    top_k = int(top_k)

    print(f"[info] config num_experts={num_experts}")
    print(f"[info] config top_k={top_k}")

    routers = find_router_modules(model, expected_num_experts=num_experts)
    if not routers:
        raise RuntimeError("No MoE router modules found. Need to inspect model.named_modules().")

    print(f"[info] found router modules: {len(routers)}")
    for name, layer_id, module in routers[:10]:
        print(f"  layer={layer_id:03d} router={name} class={type(module).__name__}")
    if len(routers) > 10:
        print(f"  ... {len(routers) - 10} more")

    counter = ExpertCounter(top_k=top_k, expected_num_experts=num_experts)

    handles = []
    for module_name, layer_id, module in routers:
        def make_hook(name: str, lid: int):
            def hook(mod, inputs, output):
                counter.add_output(name, lid, output)
            return hook

        handles.append(module.register_forward_hook(make_hook(module_name, layer_id)))

    first_device = next(model.parameters()).device
    enc = {k: v.to(first_device) for k, v in enc.items()}

    print("[info] running one prefill forward pass...")
    with torch.no_grad():
        _ = model(**enc, use_cache=False)

    for h in handles:
        h.remove()

    if not counter.stats:
        raise RuntimeError(
            "Router modules were found, but no router logits or selected expert ids were counted. "
            "Need to inspect Qwen3_5MoeTopKRouter forward output."
        )

    metadata = {
        "mode": "prefill_only",
        "model": model_name,
        "model_calls_path": str(args.model_calls),
        "call_index": args.call_index,
        "chat_template_path": chat_template_path,
        "schema_mode": args.schema_mode,
        "enable_thinking": enable_thinking,
        "rendered_prompt_path": str(out_dir / "rendered_prompt.txt"),
        "normalized_tools_path": str(out_dir / "normalized_tools.json"),
        "rendered_with_tools": used_tools,
        "num_messages": len(messages),
        "num_tools": len(tools),
        "rendered_prompt_tokens": input_tokens,
        "logged_input_tokens": logged_input_tokens,
        "token_count_delta_rendered_minus_logged": (
            None if logged_input_tokens is None else input_tokens - int(logged_input_tokens)
        ),
        "num_experts_from_config": num_experts,
        "top_k": top_k,
        "num_router_modules": len(routers),
        "router_modules": [
            {
                "name": name,
                "layer": layer_id,
                "class": type(module).__name__,
                "out_features": getattr(module, "out_features", None),
            }
            for name, layer_id, module in routers
        ],
    }

    write_outputs(out_dir=out_dir, metadata=metadata, counter=counter)

    print("[done] prefill expert counting complete.")


if __name__ == "__main__":
    main()