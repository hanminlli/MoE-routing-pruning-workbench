from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .model_structure import resolve_transformer_layers

PositionMode = Literal["all", "last", "prefill", "prefill_last", "decode"]


def _steer_tensor(
    hidden: torch.Tensor,
    *,
    direction: torch.Tensor,
    coefficient: float,
    mode: PositionMode,
) -> torch.Tensor:
    if hidden.ndim != 3:
        return hidden
    seq_len = hidden.shape[1]
    is_prefill = seq_len > 1

    apply_all = mode == "all" or (mode == "prefill" and is_prefill) or (mode == "decode" and not is_prefill)
    apply_last = mode == "last" or (mode == "prefill_last" and is_prefill)
    if not apply_all and not apply_last:
        return hidden

    vector = direction.to(device=hidden.device, dtype=hidden.dtype)
    if vector.ndim != 1 or vector.shape[0] != hidden.shape[-1]:
        raise ValueError(
            f"steering direction shape {tuple(vector.shape)} does not match hidden size {hidden.shape[-1]}"
        )

    steered = hidden.clone()
    if apply_all:
        steered = steered + coefficient * vector.view(1, 1, -1)
    elif apply_last:
        steered[:, -1, :] = steered[:, -1, :] + coefficient * vector
    return steered


def _replace_first_tensor(output: object, tensor: torch.Tensor) -> object:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (tensor, *output[1:])
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = tensor
        return output
    raise TypeError(f"Unsupported layer output type: {type(output).__name__}")


@dataclass
class ResidualSteeringHook:
    model: torch.nn.Module
    layer_index: int
    direction: torch.Tensor
    coefficient: float
    position_mode: PositionMode = "last"

    def __post_init__(self) -> None:
        layers = resolve_transformer_layers(self.model)
        if not 0 <= self.layer_index < len(layers):
            raise IndexError(f"layer_index={self.layer_index} outside 0..{len(layers)-1}")
        self._layer = layers[self.layer_index]
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        hidden = output if torch.is_tensor(output) else output[0] if isinstance(output, tuple) else output.last_hidden_state
        steered = _steer_tensor(
            hidden,
            direction=self.direction,
            coefficient=self.coefficient,
            mode=self.position_mode,
        )
        return _replace_first_tensor(output, steered)

    def __enter__(self) -> "ResidualSteeringHook":
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None
