from __future__ import annotations

import pandas as pd

from routecat_moe_steering.phase1.plans import build_plan
from routecat_moe_steering.phase1.scoring import aggregate_scores


def accounting_frame() -> pd.DataFrame:
    rows = []
    for task, tokens in [(0, 100), (1, 10), (188, 50)]:
        for layer in range(2):
            for expert in range(4):
                weighted = 0.0
                selected = 0
                if expert == 0:
                    weighted = 80.0 if task == 0 else 1.0
                    selected = 80 if task == 0 else 1
                elif expert == 1:
                    weighted = 20.0 if task == 0 else 9.0
                    selected = 20 if task == 0 else 9
                rows.append(
                    {
                        "task_num": task,
                        "layer": layer,
                        "expert": expert,
                        "selected_count": selected,
                        "weighted_count": weighted,
                        "response_tokens": tokens,
                        "sector": "A" if task == 0 else "B",
                    }
                )
    return pd.DataFrame(rows)


def test_weighted_and_task_normalized_can_rank_differently() -> None:
    frame = accounting_frame()
    weighted = aggregate_scores(
        frame,
        experiment="weighted_mass",
        excluded_tasks=(188,),
        sector=None,
    )
    normalized = aggregate_scores(
        frame,
        experiment="task_normalized_weighted_mass",
        excluded_tasks=(188,),
        sector=None,
    )
    w0 = weighted[weighted.layer == 0].sort_values("score", ascending=False).expert.iloc[0]
    n0 = normalized[normalized.layer == 0].sort_values("score", ascending=False).expert.iloc[0]
    assert w0 == 0
    assert n0 == 1


def test_sector_scoring_filters_tasks() -> None:
    scores = aggregate_scores(
        accounting_frame(),
        experiment="sector_weighted_mass",
        excluded_tasks=(188,),
        sector="B",
    )
    top = scores[scores.layer == 0].sort_values("score", ascending=False).expert.iloc[0]
    assert top == 1


def test_plan_is_deterministic_and_checkpoint_compatible() -> None:
    scores = aggregate_scores(
        accounting_frame(), experiment="weighted_mass", excluded_tasks=(188,)
    )
    plan = build_plan(
        scores,
        experiment="weighted_mass",
        keep_size=2,
        expected_layers=2,
        original_experts=4,
        top_k=1,
    )
    assert plan["retained_num_routed_experts_per_layer"] == 2
    for layer in plan["layers"].values():
        assert layer["retained_original_expert_ids"] == sorted(
            layer["retained_original_expert_ids"]
        )
        assert len(layer["original_to_new_expert_id"]) == 2
