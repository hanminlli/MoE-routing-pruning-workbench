#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from prepare_pruning_plans import (
    AGGREGATED_FILENAME,
    DEFAULT_BUCKET,
    TaskAccounting,
    discover_tasks,
    make_plan,
    parse_keep_sizes,
    parse_task_numbers,
)

CRITERION_TO_COLUMN = {
    "weighted_frequency": "weighted_count",
    "task_normalized_weighted_frequency": "task_normalized_weighted_count",
    "unweighted_frequency": "selected_count",
}

CRITERION_FORMULA = {
    "weighted_frequency": (
        "sum_over_tasks_and_response_tokens(renormalized_top8_router_probability)"
    ),
    "task_normalized_weighted_frequency": (
        "sum_over_tasks[(1/response_token_count_task) * "
        "sum_over_response_tokens(renormalized_top8_router_probability)]"
    ),
    "unweighted_frequency": (
        "sum_over_tasks_and_response_tokens(1_if_expert_is_in_selected_top8)"
    ),
}

CRITERION_AGGREGATION = {
    "weighted_frequency": "global_sum_of_weighted_top8_router_probability",
    "task_normalized_weighted_frequency": (
        "sum_of_per_task_weighted_counts_divided_by_task_response_token_count"
    ),
    "unweighted_frequency": "global_sum_of_unweighted_top8_appearances",
}


def apply_exclusions(
    tasks: list[TaskAccounting],
    inventory: pd.DataFrame,
    excluded_task_nums: set[int],
) -> tuple[list[TaskAccounting], pd.DataFrame]:
    if not excluded_task_nums:
        return tasks, inventory

    kept: list[TaskAccounting] = []
    inventory = inventory.copy()
    for task in tasks:
        if task.task_num is not None and task.task_num in excluded_task_nums:
            mask = inventory["selected"].eq(True) & inventory["task_num"].eq(task.task_num)
            inventory.loc[mask, "selected"] = False
            inventory.loc[mask, "status"] = "excluded_by_user"
            inventory.loc[mask, "reason"] = f"explicitly_excluded_task_{task.task_num:04d}"
        else:
            kept.append(task)
    return kept, inventory


def build_task_table(
    tasks: list[TaskAccounting], *, require_response_tokens: bool
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for task in tasks:
        token_count = task.total_generated_tokens
        if require_response_tokens and (token_count is None or token_count <= 0):
            raise RuntimeError(
                f"{task.task_key} has no positive total_generated_tokens; "
                "task-normalized weighted frequency cannot be computed"
            )

        frame = task.frame[["layer", "expert", "selected_count", "weighted_count"]].copy()
        frame.insert(0, "task_key", task.task_key)
        frame.insert(1, "task_num", task.task_num)
        frame["response_token_count"] = token_count
        frame["source_csv"] = str(task.csv_path)

        if token_count is not None and token_count > 0:
            frame["task_normalized_weighted_count"] = (
                frame["weighted_count"].astype(float) / float(token_count)
            )
        else:
            frame["task_normalized_weighted_count"] = np.nan
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def add_rank_and_share(scores: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    output = scores.copy().sort_values(["layer", "expert"]).reset_index(drop=True)
    totals = output.groupby("layer")[score_col].transform("sum").replace(0, np.nan)
    output[f"{prefix}_share_within_layer"] = (output[score_col] / totals).fillna(0.0)
    output[f"{prefix}_rank_within_layer"] = 0

    for layer in sorted(output["layer"].unique()):
        indices = output.index[output["layer"] == layer]
        order = output.loc[indices].sort_values(
            [score_col, "expert"], ascending=[False, True], kind="mergesort"
        ).index.tolist()
        for rank, row_index in enumerate(order, start=1):
            output.at[row_index, f"{prefix}_rank_within_layer"] = rank

    output[f"{prefix}_rank_within_layer"] = output[
        f"{prefix}_rank_within_layer"
    ].astype(int)
    return output


def criterion_coverage(
    *,
    plan_id: str,
    criterion: str,
    score_col: str,
    keep_size: int,
    plan: dict[str, Any],
    task_table: pd.DataFrame,
    global_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    retained = {
        int(layer): set(info["retained_original_expert_ids"])
        for layer, info in plan["layers"].items()
    }

    layer_rows: list[dict[str, Any]] = []
    for layer, sub in global_scores.groupby("layer", sort=True):
        mask = sub["expert"].isin(retained[int(layer)])
        total = float(sub[score_col].sum())
        kept = float(sub.loc[mask, score_col].sum())
        layer_rows.append(
            {
                "plan_id": plan_id,
                "criterion": criterion,
                "keep_size": keep_size,
                "layer": int(layer),
                "criterion_score_total": total,
                "criterion_score_retained": kept,
                "criterion_score_coverage": kept / total if total else 0.0,
            }
        )

    task_rows: list[dict[str, Any]] = []
    for task_key, sub in task_table.groupby("task_key", sort=True):
        mask = pd.Series(
            [
                int(expert) in retained[int(layer)]
                for layer, expert in zip(sub["layer"], sub["expert"])
            ],
            index=sub.index,
        )
        total = float(sub[score_col].sum())
        kept = float(sub.loc[mask, score_col].sum())
        nums = sub["task_num"].dropna().unique().tolist()
        tokens = sub["response_token_count"].dropna().unique().tolist()
        task_rows.append(
            {
                "plan_id": plan_id,
                "criterion": criterion,
                "keep_size": keep_size,
                "task_key": task_key,
                "task_num": int(nums[0]) if nums else None,
                "response_token_count": int(tokens[0]) if tokens else None,
                "criterion_score_total": total,
                "criterion_score_retained": kept,
                "criterion_score_coverage": kept / total if total else 0.0,
            }
        )

    layer_df = pd.DataFrame(layer_rows)
    task_df = pd.DataFrame(task_rows)
    total = float(global_scores[score_col].sum())
    kept = 0.0
    for layer, sub in global_scores.groupby("layer", sort=True):
        mask = sub["expert"].isin(retained[int(layer)])
        kept += float(sub.loc[mask, score_col].sum())

    summary = {
        "plan_id": plan_id,
        "criterion": criterion,
        "criterion_score_column": score_col,
        "keep_size": keep_size,
        "global_criterion_score_coverage": kept / total if total else 0.0,
        "mean_layer_criterion_score_coverage": float(
            layer_df["criterion_score_coverage"].mean()
        ),
        "min_layer_criterion_score_coverage": float(
            layer_df["criterion_score_coverage"].min()
        ),
        "mean_task_criterion_score_coverage": float(
            task_df["criterion_score_coverage"].mean()
        ),
        "p10_task_criterion_score_coverage": float(
            task_df["criterion_score_coverage"].quantile(0.10)
        ),
        "min_task_criterion_score_coverage": float(
            task_df["criterion_score_coverage"].min()
        ),
    }
    return layer_df, task_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create Experiment 1 weighted, Experiment 2 task-normalized weighted, "
            "or Experiment 3 unweighted top-8 plans from accepted response-token accounting."
        )
    )
    parser.add_argument("--accounting-root", default="accounting_result")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--criterion", required=True, choices=sorted(CRITERION_TO_COLUMN)
    )
    parser.add_argument("--keep-sizes", default="192,128,64")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--exclude-tasks", default="188")
    parser.add_argument("--aggregated-filename", default=AGGREGATED_FILENAME)
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument("--expected-experts", type=int, default=256)
    args = parser.parse_args()

    accounting_root = Path(args.accounting_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    plans_root = output_root / "plans"
    plans_root.mkdir(parents=True, exist_ok=True)

    keep_sizes = parse_keep_sizes(args.keep_sizes)
    excluded_task_nums = parse_task_numbers(args.exclude_tasks)
    score_col = CRITERION_TO_COLUMN[args.criterion]
    require_tokens = args.criterion == "task_normalized_weighted_frequency"

    tasks, inventory = discover_tasks(
        accounting_root,
        filename=args.aggregated_filename,
        bucket=args.bucket,
        expected_layers=args.expected_layers,
        expected_experts=args.expected_experts,
    )
    tasks, inventory = apply_exclusions(tasks, inventory, excluded_task_nums)
    inventory.to_csv(output_root / "task_inventory.csv", index=False)

    if not tasks:
        raise RuntimeError("No valid accounting tasks remain after exclusions")

    top_k_values = sorted({task.top_k for task in tasks})
    if len(top_k_values) != 1:
        raise RuntimeError(f"Inconsistent top_k values: {top_k_values}")
    top_k = top_k_values[0]

    task_table = build_task_table(tasks, require_response_tokens=require_tokens)
    task_table.to_csv(output_root / "task_layer_expert_scores.csv", index=False)

    global_scores = (
        task_table.groupby(["layer", "expert"], as_index=False)
        .agg(
            selected_count=("selected_count", "sum"),
            weighted_count=("weighted_count", "sum"),
            task_normalized_weighted_count=(
                "task_normalized_weighted_count",
                "sum",
            ),
            tasks_with_nonzero_selected_count=(
                "selected_count",
                lambda x: int((x > 0).sum()),
            ),
            tasks_with_nonzero_weighted_count=(
                "weighted_count",
                lambda x: int((x > 0).sum()),
            ),
        )
    )
    global_scores = add_rank_and_share(global_scores, score_col, args.criterion)
    global_scores.to_csv(output_root / "global_expert_scores.csv", index=False)

    layer_frames: list[pd.DataFrame] = []
    task_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for keep_size in keep_sizes:
        plan_id = f"{args.criterion}_keep_{keep_size:03d}"
        plan, tensor_plan = make_plan(
            criterion=args.criterion,
            score_col=score_col,
            keep_size=keep_size,
            scores=global_scores,
            num_layers=args.expected_layers,
            num_experts=args.expected_experts,
            top_k=top_k,
            bucket=args.bucket,
            accounting_root=accounting_root,
            num_tasks=len(tasks),
        )
        plan["aggregation"] = CRITERION_AGGREGATION[args.criterion]
        plan["criterion_formula"] = CRITERION_FORMULA[args.criterion]
        plan["per_task_response_token_normalization"] = require_tokens
        plan["excluded_task_nums"] = sorted(excluded_task_nums)

        (plans_root / f"{plan_id}.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(
            {
                "format_version": 1,
                "plan_id": plan_id,
                "criterion": args.criterion,
                "criterion_score_column": score_col,
                "criterion_formula": CRITERION_FORMULA[args.criterion],
                "per_task_response_token_normalization": require_tokens,
                "keep_size": keep_size,
                "bucket": args.bucket,
                "num_layers": args.expected_layers,
                "original_num_experts": args.expected_experts,
                "top_k": top_k,
                **tensor_plan,
            },
            plans_root / f"{plan_id}.pt",
        )

        layer_df, task_df, summary = criterion_coverage(
            plan_id=plan_id,
            criterion=args.criterion,
            score_col=score_col,
            keep_size=keep_size,
            plan=plan,
            task_table=task_table,
            global_scores=global_scores,
        )
        layer_frames.append(layer_df)
        task_frames.append(task_df)
        summaries.append(summary)
        print(
            f"[plan] {plan_id}: criterion coverage="
            f"{summary['global_criterion_score_coverage']:.6f}"
        )

    pd.DataFrame(summaries).to_csv(output_root / "plan_summary.csv", index=False)
    pd.concat(layer_frames, ignore_index=True).to_csv(
        output_root / "plan_layer_coverage.csv", index=False
    )
    pd.concat(task_frames, ignore_index=True).to_csv(
        output_root / "plan_task_coverage.csv", index=False
    )

    metadata = {
        "mode": "response_token_accounting_to_experiment_variant_pruning_info",
        "criterion": args.criterion,
        "criterion_score_column": score_col,
        "criterion_formula": CRITERION_FORMULA[args.criterion],
        "aggregation": CRITERION_AGGREGATION[args.criterion],
        "per_task_response_token_normalization": require_tokens,
        "accounting_root": str(accounting_root),
        "output_root": str(output_root),
        "bucket": args.bucket,
        "keep_sizes": keep_sizes,
        "excluded_task_nums": sorted(excluded_task_nums),
        "num_selected_tasks": len(tasks),
        "selected_task_keys": [task.task_key for task in tasks],
        "selected_task_nums": [
            task.task_num for task in tasks if task.task_num is not None
        ],
        "num_layers": args.expected_layers,
        "num_experts_per_layer": args.expected_experts,
        "top_k": top_k,
        "shared_expert_policy": "unchanged_not_ranked_not_pruned",
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[done] criterion: {args.criterion}")
    print(f"[done] selected calibration tasks: {len(tasks)}")
    print(f"[done] excluded tasks: {sorted(excluded_task_nums)}")
    print(f"[done] output root: {output_root}")


if __name__ == "__main__":
    main()
