#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from routecat_moe_steering.phase2.directions import SteeringArtifact
from routecat_moe_steering.phase2.hooks import ResidualSteeringHook
from routecat_moe_steering.phase2.prompting import render_prompt


def load_prompts(path: Path | None, prompt: str | None) -> list[dict[str, object]]:
    if prompt is not None:
        return [{"prompt_id": "single", "prompt": prompt}]
    if path is None:
        raise ValueError("provide --prompt or --prompts-jsonl")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("prompt")
        if not isinstance(text, str):
            raise ValueError(f"prompt row {index + 1} has no string prompt")
        rows.append({"prompt_id": row.get("prompt_id", str(index)), "prompt": text, "metadata": row})
    return rows


def generate(
    model: torch.nn.Module,
    tokenizer: object,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    embedding = model.get_input_embeddings()
    device = embedding.weight.device if embedding is not None else next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    with torch.inference_mode():
        output = model.generate(**encoded, **generation_kwargs)
    generated = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a saved steering direction to a pruned checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--coefficient", type=float, required=True)
    parser.add_argument("--position-mode", choices=["all", "last", "prefill", "prefill_last", "decode"], default="last")
    parser.add_argument("--prompt")
    parser.add_argument("--prompts-jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render each prompt with the checkpoint tokenizer chat template.",
    )
    parser.add_argument("--system-prompt")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    artifact = SteeringArtifact.load(args.artifact)
    prompts = load_prompts(Path(args.prompts_jsonl) if args.prompts_jsonl else None, args.prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in prompts:
            rendered_prompt = render_prompt(
                tokenizer,
                str(row["prompt"]),
                use_chat_template=args.use_chat_template,
                system_prompt=args.system_prompt,
            )
            with ResidualSteeringHook(
                model,
                layer_index=artifact.layer_index,
                direction=artifact.direction,
                coefficient=args.coefficient,
                position_mode=args.position_mode,
            ):
                completion = generate(
                    model,
                    tokenizer,
                    rendered_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            record = {
                **row,
                "completion": completion,
                "model": args.model,
                "artifact": str(Path(args.artifact).resolve()),
                "layer": artifact.layer_index,
                "coefficient": args.coefficient,
                "position_mode": args.position_mode,
                "use_chat_template": args.use_chat_template,
                "system_prompt": args.system_prompt,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(prompts)} steered completions to {output_path}")


if __name__ == "__main__":
    main()
