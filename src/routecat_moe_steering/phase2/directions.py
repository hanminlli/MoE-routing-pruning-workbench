from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

DirectionMethod = Literal["paired_caa", "difference_in_means"]


def l2_normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = vector.norm(p=2)
    if float(norm) <= eps:
        raise ValueError("cannot normalize a near-zero steering direction")
    return vector / norm


def discover_direction(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    method: DirectionMethod = "paired_caa",
    normalize: bool = True,
) -> torch.Tensor:
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("positive and negative activations must have shape [examples, hidden]")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative hidden sizes differ")

    if method == "paired_caa":
        if positive.shape[0] != negative.shape[0]:
            raise ValueError("paired_caa requires the same number of positive and negative examples")
        vector = (positive - negative).mean(dim=0)
    elif method == "difference_in_means":
        vector = positive.mean(dim=0) - negative.mean(dim=0)
    else:
        raise ValueError(f"unsupported direction method: {method}")

    return l2_normalize(vector) if normalize else vector


@dataclass(frozen=True)
class SteeringArtifact:
    layer_index: int
    direction: torch.Tensor
    method: str
    pooling: str
    model_reference: str
    num_positive: int
    num_negative: int
    metadata: dict[str, object]

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "layer_index": self.layer_index,
                "direction": self.direction.detach().cpu(),
                "method": self.method,
                "pooling": self.pooling,
                "model_reference": self.model_reference,
                "num_positive": self.num_positive,
                "num_negative": self.num_negative,
                "metadata": self.metadata,
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "SteeringArtifact":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            layer_index=int(payload["layer_index"]),
            direction=payload["direction"],
            method=str(payload["method"]),
            pooling=str(payload["pooling"]),
            model_reference=str(payload.get("model_reference", "")),
            num_positive=int(payload["num_positive"]),
            num_negative=int(payload["num_negative"]),
            metadata=dict(payload.get("metadata", {})),
        )
