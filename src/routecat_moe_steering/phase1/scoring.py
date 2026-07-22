from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EXPERIMENTS = {
    "weighted_mass": "weighted_count",
    "task_normalized_weighted_mass": "task_normalized_weighted_count",
    "unweighted_count": "selected_count",
    "sector_weighted_mass": "weighted_count",
}


@dataclass(frozen=True)
class ScoreContract:
    name: str
    score_column: str
    formula: str
    interpretation: str


CONTRACTS = {
    "weighted_mass": ScoreContract(
        name="weighted_mass",
        score_column="weighted_count",
        formula="sum_q sum_t p(q,t,l,e) 1{e in TopK(q,t,l)}",
        interpretation="Longer responses contribute more total router mass.",
    ),
    "task_normalized_weighted_mass": ScoreContract(
        name="task_normalized_weighted_mass",
        score_column="task_normalized_weighted_count",
        formula="sum_q (1/T_q) sum_t p(q,t,l,e) 1{e in TopK(q,t,l)}",
        interpretation="Each task contributes approximately equal total mass per layer.",
    ),
    "unweighted_count": ScoreContract(
        name="unweighted_count",
        score_column="selected_count",
        formula="sum_q sum_t 1{e in TopK(q,t,l)}",
        interpretation="Every top-k appearance receives equal weight.",
    ),
    "sector_weighted_mass": ScoreContract(
        name="sector_weighted_mass",
        score_column="weighted_count",
        formula="sum_{q in sector} sum_t p(q,t,l,e) 1{e in TopK(q,t,l)}",
        interpretation="A separate weighted-mass ranking is learned for each task sector.",
    ),
}


def normalize_task_number(value: object) -> int:
    """Convert common task identifiers such as 188, '0188', or 'task_0188' to int."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot infer task number from {value!r}")
    return int(digits[-4:])


def validate_accounting_frame(
    frame: pd.DataFrame,
    *,
    expected_layers: int = 40,
    expected_experts: int = 256,
) -> pd.DataFrame:
    required = {"task_num", "layer", "expert", "selected_count", "weighted_count"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"accounting frame is missing columns: {missing}")

    out = frame.copy()
    out["task_num"] = out["task_num"].map(normalize_task_number)
    for column in ("layer", "expert", "selected_count", "weighted_count"):
        out[column] = pd.to_numeric(out[column], errors="raise")
    out["layer"] = out["layer"].astype(int)
    out["expert"] = out["expert"].astype(int)

    if (out[["selected_count", "weighted_count"]] < 0).any().any():
        raise ValueError("routing scores must be non-negative")
    if not np.isfinite(out[["selected_count", "weighted_count"]].to_numpy(float)).all():
        raise ValueError("routing scores must be finite")

    layers = set(out["layer"].unique())
    if not layers.issubset(set(range(expected_layers))):
        raise ValueError(f"unexpected layer ids: {sorted(layers - set(range(expected_layers)))}")
    experts = set(out["expert"].unique())
    if not experts.issubset(set(range(expected_experts))):
        raise ValueError(
            f"unexpected expert ids: {sorted(experts - set(range(expected_experts)))}"
        )
    return out


def add_task_normalized_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "response_tokens" not in out.columns:
        raise ValueError("task-normalized scoring requires response_tokens")
    out["response_tokens"] = pd.to_numeric(out["response_tokens"], errors="raise")
    if (out["response_tokens"] <= 0).any():
        raise ValueError("response_tokens must be positive")
    out["task_normalized_weighted_count"] = (
        out["weighted_count"] / out["response_tokens"]
    )
    return out


def aggregate_scores(
    frame: pd.DataFrame,
    *,
    experiment: str,
    excluded_tasks: Iterable[int] = (188,),
    sector: str | None = None,
) -> pd.DataFrame:
    if experiment not in CONTRACTS:
        raise ValueError(f"unknown experiment {experiment!r}; choices={sorted(CONTRACTS)}")

    out = validate_accounting_frame(frame)
    excluded = {int(x) for x in excluded_tasks}
    out = out[~out["task_num"].isin(excluded)].copy()

    if experiment == "task_normalized_weighted_mass":
        out = add_task_normalized_scores(out)

    if experiment == "sector_weighted_mass":
        if "sector" not in out.columns:
            raise ValueError("sector_weighted_mass requires a sector column")
        if sector is None:
            raise ValueError("sector_weighted_mass requires a concrete sector")
        out = out[out["sector"].astype(str) == str(sector)].copy()
        if out.empty:
            raise ValueError(f"no accounting rows found for sector {sector!r}")

    score_column = CONTRACTS[experiment].score_column
    grouped = (
        out.groupby(["layer", "expert"], as_index=False)
        .agg(
            score=(score_column, "sum"),
            selected_count=("selected_count", "sum"),
            weighted_count=("weighted_count", "sum"),
            calibration_tasks=("task_num", "nunique"),
        )
        .sort_values(["layer", "expert"], kind="mergesort")
        .reset_index(drop=True)
    )
    return grouped


def rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    required = {"layer", "expert", "score"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"score table is missing columns: {missing}")

    frames: list[pd.DataFrame] = []
    for layer, sub in scores.groupby("layer", sort=True):
        ranked = sub.sort_values(
            ["score", "expert"], ascending=[False, True], kind="mergesort"
        ).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        total = float(ranked["score"].sum())
        ranked["share_within_layer"] = ranked["score"] / total if total > 0 else 0.0
        ranked["layer"] = int(layer)
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)
