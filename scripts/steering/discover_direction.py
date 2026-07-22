#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from routecat_moe_steering.phase2.activations import ActivationCapture
from routecat_moe_steering.phase2.directions import SteeringArtifact, discover_direction
from routecat_moe_steering.phase2.prompting import render_prompts


def read_pairs(path: Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        positive = row.get("positive", row.get("matching"))
        negative = row.get("negative", row.get("not_matching"))
        if not isinstance(positive, str) or not isinstance(negative, str):
            raise ValueError(
                f"line {line_number} must contain positive/negative or matching/not_matching strings"
            )
        pairs.append({"positive": positive, "negative": negative, "metadata": row})
    if not pairs:
        raise ValueError(f"no contrastive pairs found in {path}")
    return pairs


def encode_texts(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )


def collect(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    *,
    layer: int,
    pooling: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    embedding = model.get_input_embeddings()
    device = embedding.weight.device if embedding is not None else next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        batch = encode_texts(tokenizer, texts[start : start + batch_size], max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        with ActivationCapture(model, layer_index=layer, pooling=pooling) as capture:
            with torch.inference_mode():
                model(**batch, use_cache=False)
            pooled = capture.pooled(batch.get("attention_mask"))
        vectors.append(pooled.detach().float().cpu())
    return torch.cat(vectors, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover an activation-steering direction on a full or pruned checkpoint."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--method", choices=["paired_caa", "difference_in_means"], default="paired_caa")
    parser.add_argument("--pooling", choices=["last", "mean"], default="last")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render each contrastive text with the checkpoint tokenizer chat template.",
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

    pairs = read_pairs(Path(args.dataset))
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

    positive_texts = render_prompts(
        tokenizer,
        [row["positive"] for row in pairs],
        use_chat_template=args.use_chat_template,
        system_prompt=args.system_prompt,
    )
    negative_texts = render_prompts(
        tokenizer,
        [row["negative"] for row in pairs],
        use_chat_template=args.use_chat_template,
        system_prompt=args.system_prompt,
    )

    positive = collect(
        model,
        tokenizer,
        positive_texts,
        layer=args.layer,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    negative = collect(
        model,
        tokenizer,
        negative_texts,
        layer=args.layer,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    direction = discover_direction(positive, negative, method=args.method, normalize=True)
    artifact = SteeringArtifact(
        layer_index=args.layer,
        direction=direction,
        method=args.method,
        pooling=args.pooling,
        model_reference=args.model,
        num_positive=len(positive),
        num_negative=len(negative),
        metadata={
            "dataset": str(Path(args.dataset)),
            "max_length": args.max_length,
            "use_chat_template": args.use_chat_template,
            "system_prompt": args.system_prompt,
        },
    )
    output = artifact.save(args.output)
    print(f"[done] steering artifact: {output}")
    print(f"[info] direction_norm={float(direction.norm()):.6f}")


if __name__ == "__main__":
    main()
