from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def resolve_transformer_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    """Resolve decoder blocks for Qwen-style and common Hugging Face causal LMs."""
    candidates: list[Any] = [
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
    ]
    for value in candidates:
        if isinstance(value, (torch.nn.ModuleList, list, tuple)) and value:
            return value
    raise ValueError(
        "Could not locate transformer layers. Supported layouts include "
        "model.language_model.layers, model.layers, transformer.h, and gpt_neox.layers."
    )


def hidden_size_from_model(model: torch.nn.Module) -> int:
    configs = [getattr(model, "config", None)]
    base = getattr(model, "model", None)
    configs.append(getattr(base, "config", None))
    for config in configs:
        if config is None:
            continue
        for name in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
    raise ValueError("Could not infer hidden size from model config")
