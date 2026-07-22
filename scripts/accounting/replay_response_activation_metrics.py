from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM


MODE = "response_activation_nine_statistics_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# RESPONSE_ACTIVATION_JSON_STREAM_REPAIR_V1

def _escape_raw_control_characters_inside_json_strings(text: str) -> tuple[str, int]:
    """Repair physical control characters embedded inside JSON strings."""
    out: list[str] = []
    in_string = False
    escaped = False
    repaired = 0
    short_escapes = {
        "\b": r"\b",
        "\t": r"\t",
        "\n": r"\n",
        "\f": r"\f",
        "\r": r"\r",
    }

    for ch in text:
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
                escaped = False
            continue

        if escaped:
            out.append(ch)
            escaped = False
            continue

        if ch == "\\":
            out.append(ch)
            escaped = True
            continue

        if ch == '"':
            out.append(ch)
            in_string = False
            continue

        if ord(ch) < 0x20:
            out.append(short_escapes.get(ch, f"\\u{ord(ch):04x}"))
            repaired += 1
            continue

        out.append(ch)

    if in_string:
        raise ValueError("model_calls file ends inside an unterminated JSON string")

    return "".join(out), repaired


def _decode_json_object_stream(text: str, *, source: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    length = len(text)

    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break

        try:
            obj, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            left = max(0, exc.pos - 120)
            right = min(length, exc.pos + 120)
            context = text[left:right].replace("\n", r"\n").replace("\r", r"\r")
            raise ValueError(
                f"could not parse JSON object stream {source} at character {exc.pos}: "
                f"{exc.msg}; nearby={context!r}"
            ) from exc

        if not isinstance(obj, dict):
            raise ValueError(
                f"expected a JSON object in {source} at character {index}, "
                f"got {type(obj).__name__}"
            )

        objects.append(obj)
        index = end

    return objects


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8-sig")

    try:
        calls = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if all(isinstance(call, dict) for call in calls):
            return calls
    except json.JSONDecodeError:
        pass

    repaired_text, repaired_count = _escape_raw_control_characters_inside_json_strings(raw)
    calls = _decode_json_object_stream(repaired_text, source=str(path))
    print(
        f"[warn] compatibility JSON parser used for {path}; "
        f"repaired_raw_control_characters={repaired_count}; parsed_calls={len(calls)}"
    )
    return calls


def parse_indices(spec: str, available: list[int]) -> list[int]:
    if spec == "all":
        return available

    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))

    wanted = sorted(dict.fromkeys(out))
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise ValueError(f"requested call indices are absent: {missing}")
    return wanted


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


def text_config(config: Any) -> Any:
    nested = getattr(config, "text_config", None)
    return nested if nested is not None else config


def first_attr(config: Any, names: list[str]) -> Any:
    for candidate in (text_config(config), config):
        for name in names:
            value = getattr(candidate, name, None)
            if value is not None:
                return value
    return None


def parse_layer_id(module_name: str) -> int:
    match = re.search(r"layers\.(\d+)", module_name)
    if not match:
        raise ValueError(f"cannot parse layer id from module name: {module_name}")
    return int(match.group(1))


def find_expert_modules(model: torch.nn.Module) -> list[tuple[str, int, torch.nn.Module]]:
    found: list[tuple[str, int, torch.nn.Module]] = []

    for name, module in model.named_modules():
        if not name.endswith(".mlp.experts"):
            continue
        if not hasattr(module, "gate_up_proj") or not hasattr(module, "down_proj"):
            continue
        if not hasattr(module, "act_fn"):
            continue
        found.append((name, parse_layer_id(name), module))

    return sorted(found, key=lambda x: x[1])


def model_input_device(model: torch.nn.Module) -> torch.device:
    embedding = model.get_input_embeddings()
    if embedding is not None and hasattr(embedding, "weight"):
        return embedding.weight.device
    return next(model.parameters()).device


def select_forward_model(model: torch.nn.Module) -> torch.nn.Module:
    for attr in ("model", "transformer", "base_model"):
        candidate = getattr(model, attr, None)
        if candidate is not None and candidate is not model:
            return candidate
    return model


class ResponseActivationCollector:
    """Collect A^(alpha,beta) for alpha,beta in {0,1,2}.

    For a selected expert event:
        g = normalized top-k router weight
        r = L2 norm of the raw expert output before multiplying by g

    The stored score is:
        A^(alpha,beta) += g^alpha * r^beta
    """

    def __init__(
        self,
        *,
        response_positions: list[int],
        num_experts: int,
        top_k: int,
        activation_chunk_size: int,
    ) -> None:
        self.response_positions = response_positions
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.activation_chunk_size = int(activation_chunk_size)

        # CPU float64 is small and stable: [layer, expert, alpha, beta].
        self.scores: dict[int, torch.Tensor] = {}
        self.max_activation_l2: dict[int, torch.Tensor] = {}
        self.layer_module_names: dict[int, str] = {}
        self.layer_response_tokens: dict[int, int] = {}

    @torch.inference_mode()
    def add_expert_call(
        self,
        module_name: str,
        layer_id: int,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
    ) -> None:
        if len(inputs) < 3:
            raise RuntimeError(
                f"{module_name} hook expected at least 3 positional inputs "
                f"(hidden_states, top_k_index, top_k_weights), got {len(inputs)}"
            )

        hidden_states, top_k_index, top_k_weights = inputs[:3]
        if not (torch.is_tensor(hidden_states) and torch.is_tensor(top_k_index) and torch.is_tensor(top_k_weights)):
            raise TypeError(f"{module_name} received non-tensor expert inputs")

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        top_k_index = top_k_index.reshape(-1, top_k_index.shape[-1])
        top_k_weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])

        seq_len = int(hidden_states.shape[0])
        if top_k_index.shape[0] != seq_len or top_k_weights.shape[0] != seq_len:
            raise RuntimeError(
                f"{module_name} inconsistent token dimension: hidden={seq_len}, "
                f"indices={top_k_index.shape[0]}, weights={top_k_weights.shape[0]}"
            )
        if int(top_k_index.shape[-1]) != self.top_k:
            raise RuntimeError(
                f"{module_name} top-k mismatch: observed={top_k_index.shape[-1]}, expected={self.top_k}"
            )

        valid_positions = [p for p in self.response_positions if 0 <= p < seq_len]
        if len(valid_positions) != len(self.response_positions):
            raise RuntimeError(
                f"{module_name} response positions exceed sequence length: "
                f"positions={len(self.response_positions)}, valid={len(valid_positions)}, seq_len={seq_len}"
            )
        if not valid_positions:
            return

        device = hidden_states.device
        position_tensor = torch.tensor(valid_positions, dtype=torch.long, device=device)
        response_hidden = hidden_states.index_select(0, position_tensor)
        response_experts = top_k_index.index_select(0, position_tensor).long()
        response_weights = top_k_weights.index_select(0, position_tensor).float()

        if int(response_experts.min().item()) < 0 or int(response_experts.max().item()) >= self.num_experts:
            raise RuntimeError(f"{module_name} produced expert id outside [0, {self.num_experts - 1}]")

        layer_scores = torch.zeros(
            (self.num_experts, 3, 3),
            dtype=torch.float64,
            device="cpu",
        )
        layer_max = torch.zeros((self.num_experts,), dtype=torch.float64, device="cpu")

        selected_experts = torch.unique(response_experts).tolist()
        for expert_id_raw in selected_experts:
            expert_id = int(expert_id_raw)
            token_idx, top_k_pos = torch.where(response_experts == expert_id)
            num_events = int(token_idx.numel())
            if num_events == 0:
                continue

            event_scores = torch.zeros((3, 3), dtype=torch.float64, device="cpu")
            event_max = 0.0

            for start in range(0, num_events, self.activation_chunk_size):
                end = min(start + self.activation_chunk_size, num_events)
                token_chunk = token_idx[start:end]
                rank_chunk = top_k_pos[start:end]

                current_state = response_hidden.index_select(0, token_chunk)
                gate, up = F.linear(current_state, module.gate_up_proj[expert_id]).chunk(2, dim=-1)
                raw_expert_output = F.linear(module.act_fn(gate) * up, module.down_proj[expert_id])

                activation_l2 = torch.linalg.vector_norm(raw_expert_output.float(), ord=2, dim=-1)
                gate_weight = response_weights[token_chunk, rank_chunk]

                gate_powers = torch.stack(
                    (
                        torch.ones_like(gate_weight),
                        gate_weight,
                        gate_weight.square(),
                    ),
                    dim=1,
                )
                activation_powers = torch.stack(
                    (
                        torch.ones_like(activation_l2),
                        activation_l2,
                        activation_l2.square(),
                    ),
                    dim=1,
                )

                # [3, 3], summing over selected routed events.
                chunk_scores = torch.einsum("na,nb->ab", gate_powers, activation_powers)
                event_scores += chunk_scores.double().cpu()
                event_max = max(event_max, float(activation_l2.max().item()))

                del current_state, gate, up, raw_expert_output
                del activation_l2, gate_weight, gate_powers, activation_powers, chunk_scores

            layer_scores[expert_id] += event_scores
            layer_max[expert_id] = event_max

        self.scores[layer_id] = layer_scores
        self.max_activation_l2[layer_id] = layer_max
        self.layer_module_names[layer_id] = module_name
        self.layer_response_tokens[layer_id] = len(valid_positions)

        del position_tensor, response_hidden, response_experts, response_weights


def write_call_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "call_index",
        "prompt_tokens",
        "generated_tokens",
        "full_sequence_tokens",
        "logged_input_tokens",
        "logged_answer_tokens",
        "logged_reasoning_tokens",
        "logged_generated_tokens",
        "prompt_delta",
        "generated_delta",
        "response_prediction_positions",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_expert_csv(
    path: Path,
    *,
    total_scores: torch.Tensor,
    max_activation_l2: torch.Tensor,
    layer_ids: list[int],
    module_names: dict[int, str],
    total_response_tokens: int,
    top_k: int,
) -> None:
    fieldnames = [
        "layer",
        "module_name",
        "expert",
        "response_tokens",
        "top_k",
        "a0_b0_routed_count",
        "a0_b1_activation_l2_sum",
        "a0_b2_activation_l2_sq_sum",
        "a1_b0_gate_sum",
        "a1_b1_gate_x_activation_l2_sum",
        "a1_b2_gate_x_activation_l2_sq_sum",
        "a2_b0_gate_sq_sum",
        "a2_b1_gate_sq_x_activation_l2_sum",
        "a2_b2_gate_sq_x_activation_l2_sq_sum",
        "mean_gate_when_selected",
        "man_mean_activation_l2",
        "msan_mean_activation_l2_sq",
        "activation_l2_max",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for layer_offset, layer_id in enumerate(layer_ids):
            for expert_id in range(total_scores.shape[1]):
                s = total_scores[layer_offset, expert_id]
                count = float(s[0, 0].item())
                gate_sum = float(s[1, 0].item())
                activation_sum = float(s[0, 1].item())
                activation_sq_sum = float(s[0, 2].item())

                writer.writerow(
                    {
                        "layer": layer_id,
                        "module_name": module_names[layer_id],
                        "expert": expert_id,
                        "response_tokens": total_response_tokens,
                        "top_k": top_k,
                        "a0_b0_routed_count": int(round(count)),
                        "a0_b1_activation_l2_sum": activation_sum,
                        "a0_b2_activation_l2_sq_sum": activation_sq_sum,
                        "a1_b0_gate_sum": gate_sum,
                        "a1_b1_gate_x_activation_l2_sum": float(s[1, 1].item()),
                        "a1_b2_gate_x_activation_l2_sq_sum": float(s[1, 2].item()),
                        "a2_b0_gate_sq_sum": float(s[2, 0].item()),
                        "a2_b1_gate_sq_x_activation_l2_sum": float(s[2, 1].item()),
                        "a2_b2_gate_sq_x_activation_l2_sq_sum": float(s[2, 2].item()),
                        "mean_gate_when_selected": gate_sum / count if count > 0 else 0.0,
                        "man_mean_activation_l2": activation_sum / count if count > 0 else 0.0,
                        "msan_mean_activation_l2_sq": activation_sq_sum / count if count > 0 else 0.0,
                        "activation_l2_max": float(max_activation_l2[layer_offset, expert_id].item()),
                    }
                )


def write_layer_checks(
    path: Path,
    *,
    total_scores: torch.Tensor,
    layer_ids: list[int],
    module_names: dict[int, str],
    total_response_tokens: int,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_count = total_response_tokens * top_k
    expected_gate_sum = float(total_response_tokens)

    for layer_offset, layer_id in enumerate(layer_ids):
        observed_count = int(round(float(total_scores[layer_offset, :, 0, 0].sum().item())))
        observed_gate_sum = float(total_scores[layer_offset, :, 1, 0].sum().item())
        row = {
            "layer": layer_id,
            "module_name": module_names[layer_id],
            "response_tokens": total_response_tokens,
            "top_k": top_k,
            "observed_routed_count": observed_count,
            "expected_routed_count": expected_count,
            "routed_count_delta": observed_count - expected_count,
            "routed_count_ok": observed_count == expected_count,
            "observed_gate_sum": observed_gate_sum,
            "expected_gate_sum": expected_gate_sum,
            "gate_sum_delta": observed_gate_sum - expected_gate_sum,
            "gate_sum_ok": math.isclose(observed_gate_sum, expected_gate_sum, rel_tol=5e-5, abs_tol=1e-2),
        }
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run_config.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-calls", required=True)
    parser.add_argument("--call-indices", default="all")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--activation-chunk-size", type=int, default=1024)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.activation_chunk_size <= 0:
        raise ValueError("--activation-chunk-size must be positive")

    cfg = load_json(Path(args.config)) if Path(args.config).exists() else {}
    model_name = args.model or cfg.get("model")
    if not model_name:
        raise ValueError("Model name missing. Provide --model or config['model'].")

    calls_all = iter_jsonl(Path(args.model_calls))
    by_index = {int(call["call_index"]): call for call in calls_all}
    wanted = parse_indices(args.call_indices, sorted(by_index))
    calls = [by_index[i] for i in wanted]

    print(f"[info] transformers={transformers.__version__}")
    print(f"[info] torch={torch.__version__}")
    print(f"[info] loading model once: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.eval()

    forward_model = select_forward_model(model)
    if forward_model is model:
        print("[warn] using the full CausalLM model; LM-head logits may consume extra memory")
    else:
        print(f"[info] using base transformer for replay: {type(forward_model).__name__}")

    num_experts_value = first_attr(
        model.config,
        ["num_experts", "n_routed_experts", "num_routed_experts", "moe_num_experts"],
    )
    top_k_value = args.top_k or first_attr(
        model.config,
        ["num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k", "router_top_k"],
    )
    if num_experts_value is None:
        raise ValueError("Could not infer num_experts from model config")
    if top_k_value is None:
        raise ValueError("Could not infer top_k from model config; pass --top-k")

    num_experts = int(num_experts_value)
    top_k = int(top_k_value)

    expert_modules = find_expert_modules(model)
    if not expert_modules:
        raise RuntimeError("No routed expert modules ending in '.mlp.experts' were found")

    layer_ids = [layer_id for _, layer_id, _ in expert_modules]
    if len(set(layer_ids)) != len(layer_ids):
        raise RuntimeError(f"duplicate routed layer ids found: {layer_ids}")

    module_names = {layer_id: name for name, layer_id, _ in expert_modules}
    print(
        f"[info] routed_layers={len(expert_modules)} num_experts={num_experts} "
        f"top_k={top_k} activation_chunk_size={args.activation_chunk_size}"
    )
    print(
        f"[info] expert forward signature: "
        f"{inspect.signature(expert_modules[0][2].forward)}"
    )

    total_scores = torch.zeros(
        (len(layer_ids), num_experts, 3, 3),
        dtype=torch.float64,
        device="cpu",
    )
    max_activation_l2 = torch.zeros(
        (len(layer_ids), num_experts),
        dtype=torch.float64,
        device="cpu",
    )
    layer_to_offset = {layer_id: offset for offset, layer_id in enumerate(layer_ids)}

    current_collector: ResponseActivationCollector | None = None
    handles = []

    for module_name, layer_id, module in expert_modules:
        def make_hook(name: str, lid: int):
            def hook(mod: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                if current_collector is not None:
                    current_collector.add_expert_call(name, lid, mod, inputs)
            return hook

        handles.append(module.register_forward_pre_hook(make_hook(module_name, layer_id)))

    call_rows: list[dict[str, Any]] = []
    total_response_tokens = 0
    input_device = model_input_device(model)

    try:
        for call in calls:
            call_index = int(call["call_index"])
            prompt_ids, generated_ids = get_token_ids(call)
            n_prompt = len(prompt_ids)
            n_generated = len(generated_ids)

            if n_prompt <= 0:
                raise ValueError(f"call_index={call_index} has an empty prompt")

            usage = call.get("response", {}).get("token_usage", {})
            logged_input = int(usage.get("input", 0) or 0)
            logged_answer = int(usage.get("answer", 0) or 0)
            logged_reasoning = int(usage.get("reasoning", 0) or 0)
            logged_generated = logged_answer + logged_reasoning

            if n_prompt != logged_input:
                raise ValueError(
                    f"call_index={call_index}: prompt ids length {n_prompt} != logged input {logged_input}"
                )
            if n_generated != logged_generated:
                raise ValueError(
                    f"call_index={call_index}: generated ids length {n_generated} "
                    f"!= logged generated {logged_generated}"
                )

            full_ids = prompt_ids + generated_ids
            response_positions = (
                list(range(n_prompt - 1, n_prompt + n_generated - 1))
                if n_generated > 0
                else []
            )

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
                    "response_prediction_positions": len(response_positions),
                }
            )

            print(
                f"[info] call={call_index} prompt={n_prompt} generated={n_generated} "
                f"full={len(full_ids)}"
            )

            if n_generated == 0:
                continue

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=input_device)
            attention_mask = torch.ones_like(input_ids, device=input_device)

            current_collector = ResponseActivationCollector(
                response_positions=response_positions,
                num_experts=num_experts,
                top_k=top_k,
                activation_chunk_size=args.activation_chunk_size,
            )

            with torch.inference_mode():
                output = forward_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=False,
                )

            missing_layers = sorted(set(layer_ids) - set(current_collector.scores))
            if missing_layers:
                raise RuntimeError(
                    f"call_index={call_index}: expert hooks did not fire for layers {missing_layers}"
                )

            for layer_id in layer_ids:
                offset = layer_to_offset[layer_id]
                total_scores[offset] += current_collector.scores[layer_id]
                max_activation_l2[offset] = torch.maximum(
                    max_activation_l2[offset],
                    current_collector.max_activation_l2[layer_id],
                )

            total_response_tokens += n_generated
            current_collector = None

            del output, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        current_collector = None
        for handle in handles:
            handle.remove()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_call_summary(out_dir / "call_token_summary.csv", call_rows)
    write_expert_csv(
        out_dir / "expert_response_activation_aggregated.csv",
        total_scores=total_scores,
        max_activation_l2=max_activation_l2,
        layer_ids=layer_ids,
        module_names=module_names,
        total_response_tokens=total_response_tokens,
        top_k=top_k,
    )
    layer_rows = write_layer_checks(
        out_dir / "layer_conservation.csv",
        total_scores=total_scores,
        layer_ids=layer_ids,
        module_names=module_names,
        total_response_tokens=total_response_tokens,
        top_k=top_k,
    )

    all_scores_finite = bool(torch.isfinite(total_scores).all().item())
    all_scores_nonnegative = bool((total_scores >= 0).all().item())

    observed_total_count = int(round(float(total_scores[:, :, 0, 0].sum().item())))
    expected_total_count = total_response_tokens * top_k * len(layer_ids)
    observed_total_gate_sum = float(total_scores[:, :, 1, 0].sum().item())
    expected_total_gate_sum = float(total_response_tokens * len(layer_ids))

    response_count_matches = observed_total_count == expected_total_count
    gate_sum_matches = math.isclose(
        observed_total_gate_sum,
        expected_total_gate_sum,
        rel_tol=5e-5,
        abs_tol=max(1e-2, 1e-6 * expected_total_gate_sum),
    )
    all_layer_count_checks = all(bool(row["routed_count_ok"]) for row in layer_rows)
    all_layer_gate_checks = all(bool(row["gate_sum_ok"]) for row in layer_rows)

    state = {
        "mode": MODE,
        "model": model_name,
        "model_calls_path": str(args.model_calls),
        "call_indices": wanted,
        "layer_ids": torch.tensor(layer_ids, dtype=torch.int64),
        "module_names": [module_names[layer_id] for layer_id in layer_ids],
        "num_experts": num_experts,
        "top_k": top_k,
        "total_response_tokens": total_response_tokens,
        "scores_alpha_beta": total_scores,
        "activation_l2_max": max_activation_l2,
        "axis_meaning": {
            "scores_alpha_beta": "[layer, expert, alpha, beta]",
            "alpha": [0, 1, 2],
            "beta": [0, 1, 2],
            "score": "sum over selected response-token routed events of gate_weight**alpha * raw_expert_output_l2**beta",
        },
    }
    torch.save(state, out_dir / "observer_state.pt")

    metadata = {
        "mode": MODE,
        "meaning": {
            "positions": "generated-output prediction positions N-1..N+T-2, one position per logged generated token",
            "activation": "L2 norm of each selected routed expert output before multiplying by its normalized top-k router weight",
            "score": "A^(alpha,beta) = sum gate_weight^alpha * activation_l2^beta for alpha,beta in {0,1,2}",
            "shared_expert": "not included; only modules ending in .mlp.experts are observed",
        },
        "model": model_name,
        "model_calls_path": str(args.model_calls),
        "call_indices": wanted,
        "num_calls": len(wanted),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "forward_model_class": type(forward_model).__name__,
        "expert_module_class": type(expert_modules[0][2]).__name__,
        "num_router_layers": len(layer_ids),
        "layer_ids": layer_ids,
        "num_experts": num_experts,
        "top_k": top_k,
        "activation_chunk_size": args.activation_chunk_size,
        "total_response_tokens": total_response_tokens,
        "observed_total_routed_count": observed_total_count,
        "expected_total_routed_count": expected_total_count,
        "response_count_matches_expected_total": response_count_matches,
        "observed_total_gate_sum": observed_total_gate_sum,
        "expected_total_gate_sum": expected_total_gate_sum,
        "gate_sum_delta": observed_total_gate_sum - expected_total_gate_sum,
        "gate_sum_matches_expected_total": gate_sum_matches,
        "all_layer_count_checks_pass": all_layer_count_checks,
        "all_layer_gate_checks_pass": all_layer_gate_checks,
        "all_scores_finite": all_scores_finite,
        "all_scores_nonnegative": all_scores_nonnegative,
        "outputs": {
            "call_token_summary": "call_token_summary.csv",
            "expert_aggregated": "expert_response_activation_aggregated.csv",
            "layer_conservation": "layer_conservation.csv",
            "observer_state": "observer_state.pt",
        },
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[done] wrote:")
    for filename in (
        "metadata.json",
        "call_token_summary.csv",
        "expert_response_activation_aggregated.csv",
        "layer_conservation.csv",
        "observer_state.pt",
    ):
        print(" ", out_dir / filename)

    print()
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    if not (
        response_count_matches
        and gate_sum_matches
        and all_layer_count_checks
        and all_layer_gate_checks
        and all_scores_finite
        and all_scores_nonnegative
    ):
        raise SystemExit("One or more response-activation accounting checks failed")


if __name__ == "__main__":
    main()
