from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .scoring import CONTRACTS, rank_scores


def build_plan(
    scores: pd.DataFrame,
    *,
    experiment: str,
    keep_size: int,
    expected_layers: int = 40,
    original_experts: int = 256,
    top_k: int = 8,
    source: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if experiment not in CONTRACTS:
        raise ValueError(f"unknown experiment: {experiment}")
    if not top_k <= keep_size <= original_experts:
        raise ValueError(
            f"keep_size must satisfy top_k <= keep_size <= original_experts; "
            f"got {top_k}, {keep_size}, {original_experts}"
        )

    ranked = rank_scores(scores)
    expected_ids = set(range(original_experts))
    layers: dict[str, Any] = {}

    for layer in range(expected_layers):
        sub = ranked[ranked["layer"] == layer]
        if set(sub["expert"].astype(int)) != expected_ids:
            raise ValueError(f"layer {layer} does not contain experts 0..{original_experts - 1}")

        retained_by_score = sub.head(keep_size)["expert"].astype(int).tolist()
        retained_sorted = sorted(retained_by_score)
        retained_set = set(retained_sorted)
        pruned = [expert for expert in range(original_experts) if expert not in retained_set]
        score_by_id = dict(zip(sub["expert"].astype(int), sub["score"].astype(float)))
        all_scores = [float(score_by_id[e]) for e in range(original_experts)]
        total = float(sum(all_scores))
        kept = float(sum(score_by_id[e] for e in retained_sorted))

        layers[str(layer)] = {
            "layer": layer,
            "keep_size": keep_size,
            "retained_original_expert_ids": retained_sorted,
            "retained_original_expert_ids_by_score": retained_by_score,
            "pruned_original_expert_ids": pruned,
            "new_to_original_expert_id": retained_sorted,
            "original_to_new_expert_id": {
                str(original): new for new, original in enumerate(retained_sorted)
            },
            "criterion_score_by_original_expert_id": all_scores,
            "criterion_score_total": total,
            "criterion_score_retained": kept,
            "criterion_score_coverage": kept / total if total else 0.0,
        }

    contract = CONTRACTS[experiment]
    plan: dict[str, Any] = {
        "format_version": 1,
        "plan_type": "static_per_layer_routed_expert_pruning",
        "experiment": experiment,
        "criterion": contract.name,
        "criterion_formula": contract.formula,
        "criterion_interpretation": contract.interpretation,
        "source": source,
        "num_layers": expected_layers,
        "original_num_routed_experts_per_layer": original_experts,
        "retained_num_routed_experts_per_layer": keep_size,
        "pruned_num_routed_experts_per_layer": original_experts - keep_size,
        "num_experts_per_token_top_k": top_k,
        "shared_expert_policy": "unchanged_not_ranked_not_pruned",
        "tie_break_rule": "score_descending_then_original_expert_id_ascending",
        "checkpoint_reindex_rule": (
            "Slice each layer using retained_original_expert_ids sorted by original ID; "
            "the new expert ID is its position in that sorted list."
        ),
        "metadata": metadata or {},
        "layers": layers,
    }
    return plan


def write_plan(path: str | Path, plan: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
