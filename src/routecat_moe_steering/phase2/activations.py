from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .model_structure import resolve_transformer_layers

Pooling = Literal["last", "mean"]


def _first_tensor(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state
    raise TypeError(f"Unsupported layer output type: {type(output).__name__}")


@dataclass
class ActivationCapture:
    model: torch.nn.Module
    layer_index: int
    pooling: Pooling = "last"

    def __post_init__(self) -> None:
        layers = resolve_transformer_layers(self.model)
        if not 0 <= self.layer_index < len(layers):
            raise IndexError(f"layer_index={self.layer_index} outside 0..{len(layers)-1}")
        self._layer = layers[self.layer_index]
        self._captured: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        self._captured = _first_tensor(output).detach()

    def __enter__(self) -> "ActivationCapture":
        self._captured = None
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None

    def pooled(self, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self._captured is None:
            raise RuntimeError("No activation was captured; run a model forward pass first")
        hidden = self._captured
        if hidden.ndim != 3:
            raise ValueError(f"expected [batch, sequence, hidden], got {tuple(hidden.shape)}")

        if self.pooling == "last":
            if attention_mask is None:
                return hidden[:, -1, :]
            lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[batch, lengths, :]

        if self.pooling == "mean":
            if attention_mask is None:
                return hidden.mean(dim=1)
            weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        raise ValueError(f"unknown pooling mode: {self.pooling}")
