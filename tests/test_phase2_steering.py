from __future__ import annotations

import torch

from routecat_moe_steering.phase2.directions import discover_direction
from routecat_moe_steering.phase2.hooks import ResidualSteeringHook, _steer_tensor


class Block(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x + 1,)


class Inner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(), Block()])


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


def test_paired_caa_direction_is_normalized() -> None:
    positive = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
    negative = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    direction = discover_direction(positive, negative, method="paired_caa")
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]))
    assert torch.allclose(direction.norm(), torch.tensor(1.0))


def test_last_position_steering() -> None:
    hidden = torch.zeros(1, 3, 2)
    out = _steer_tensor(
        hidden,
        direction=torch.tensor([1.0, -1.0]),
        coefficient=2.0,
        mode="last",
    )
    assert torch.allclose(out[:, :2], torch.zeros(1, 2, 2))
    assert torch.allclose(out[:, 2], torch.tensor([[2.0, -2.0]]))


def test_hook_changes_selected_layer_output() -> None:
    model = DummyModel()
    x = torch.zeros(1, 2, 3)
    baseline = model(x)
    with ResidualSteeringHook(
        model,
        layer_index=0,
        direction=torch.tensor([1.0, 0.0, 0.0]),
        coefficient=3.0,
        position_mode="all",
    ):
        steered = model(x)
    assert torch.allclose(baseline, torch.full_like(x, 2.0))
    assert torch.allclose(steered[..., 0], torch.full((1, 2), 5.0))
    assert torch.allclose(steered[..., 1:], torch.full((1, 2, 2), 2.0))

class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return "|".join(f"{row['role']}:{row['content']}" for row in messages) + "|assistant:"


def test_prompt_rendering_uses_checkpoint_chat_template() -> None:
    from routecat_moe_steering.phase2.prompting import render_prompt

    rendered = render_prompt(
        FakeTokenizer(),
        "Explain the result.",
        system_prompt="Be precise.",
    )
    assert rendered == "system:Be precise.|user:Explain the result.|assistant:"


def test_prompt_rendering_can_preserve_raw_text() -> None:
    from routecat_moe_steering.phase2.prompting import render_prompt

    assert render_prompt(FakeTokenizer(), "already rendered", use_chat_template=False) == "already rendered"
