#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

AGGREGATED_FILENAME = "expert_counts_prompt_and_generated_aggregated.csv"
DEFAULT_BUCKET = "generated_output_prediction"


@dataclass(frozen=True)
class TaskAccounting:
    task_key: str
    task_num: int | None
    csv_path: Path
    frame: pd.DataFrame
    top_k: int
    total_generated_tokens: int | None


def parse_task_numbers(spec: str) -> set[int]:
    """Parse comma-separated task numbers and inclusive ranges.

    Examples:
        "188"
        "12,18,188"
        "50-60,188"
    """
    out: set[int] = set()

    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue

        if "-" in raw:
            left, right = raw.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(
                    f"invalid task range {raw!r}: end is smaller than start"
                )
            out.update(range(start, end + 1))
        else:
            out.add(int(raw))

    return out


def parse_keep_sizes(spec: str) -> list[int]:
    values = sorted({int(x.strip()) for x in spec.split(",") if x.strip()}, reverse=True)
    if not values or any(x <= 0 for x in values):
        raise ValueError("--keep-sizes must contain positive integers")
    return values


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def infer_task(path: Path) -> tuple[str, int | None]:
    matches = re.findall(r"task[_-](\d{4})", str(path))
    if matches:
        task_num = int(matches[-1])
        return f"task_{task_num:04d}", task_num
    return path.parent.name, None


def validate_one_csv(
    csv_path: Path,
    *,
    bucket: str,
    expected_layers: int,
    expected_experts: int,
) -> tuple[TaskAccounting | None, dict[str, Any]]:
    task_key, task_num = infer_task(csv_path)
    metadata_path = csv_path.parent / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else None

    inventory: dict[str, Any] = {
        "task_key": task_key,
        "task_num": task_num,
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path) if metadata_path.exists() else "",
        "mtime": csv_path.stat().st_mtime,
        "selected": False,
        "status": "",
        "reason": "",
    }

    if metadata is not None and metadata.get("generated_matches_expected_total") is False:
        inventory.update(status="rejected", reason="metadata_generated_conservation_failed")
        return None, inventory

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        inventory.update(status="rejected", reason=f"csv_read_error:{type(exc).__name__}:{exc}")
        return None, inventory

    required = {"bucket", "layer", "expert", "selected_count", "weighted_count"}
    missing = sorted(required - set(df.columns))
    if missing:
        inventory.update(status="rejected", reason=f"missing_columns:{','.join(missing)}")
        return None, inventory

    df = df[df["bucket"].astype(str) == bucket].copy()
    if df.empty:
        inventory.update(status="rejected", reason=f"bucket_not_found:{bucket}")
        return None, inventory

    numeric_cols = ["layer", "expert", "selected_count", "weighted_count"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric_cols].isna().any().any():
        inventory.update(status="rejected", reason="missing_or_nonnumeric_values")
        return None, inventory

    values = df[["selected_count", "weighted_count"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        inventory.update(status="rejected", reason="invalid_negative_or_nonfinite_scores")
        return None, inventory

    df["layer"] = df["layer"].astype(int)
    df["expert"] = df["expert"].astype(int)

    aggregation: dict[str, str] = {
        "selected_count": "sum",
        "weighted_count": "sum",
    }
    if "tokens" in df.columns:
        df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce")
        aggregation["tokens"] = "max"
    if "top_k" in df.columns:
        df["top_k"] = pd.to_numeric(df["top_k"], errors="coerce")
        aggregation["top_k"] = "max"

    df = (
        df.groupby(["layer", "expert"], as_index=False)
        .agg(aggregation)
        .sort_values(["layer", "expert"])
        .reset_index(drop=True)
    )

    expected_rows = expected_layers * expected_experts
    if len(df) != expected_rows:
        inventory.update(status="rejected", reason=f"row_count_{len(df)}_expected_{expected_rows}")
        return None, inventory

    if set(df["layer"].unique()) != set(range(expected_layers)):
        inventory.update(status="rejected", reason="incomplete_layer_ids")
        return None, inventory

    expected_expert_ids = set(range(expected_experts))
    for layer in range(expected_layers):
        ids = set(df.loc[df["layer"] == layer, "expert"].tolist())
        if ids != expected_expert_ids:
            inventory.update(status="rejected", reason=f"incomplete_expert_ids_layer_{layer}")
            return None, inventory

    top_k = 8
    if "top_k" in df.columns and df["top_k"].notna().any():
        values = sorted(set(df["top_k"].dropna().astype(int).tolist()))
        if len(values) != 1:
            inventory.update(status="rejected", reason=f"inconsistent_top_k:{values}")
            return None, inventory
        top_k = values[0]
    elif metadata and metadata.get("top_k") is not None:
        top_k = int(metadata["top_k"])

    total_generated_tokens: int | None = None
    if metadata and metadata.get("total_generated_tokens") is not None:
        total_generated_tokens = int(metadata["total_generated_tokens"])
    elif "tokens" in df.columns:
        per_layer = df.groupby("layer")["tokens"].max().dropna().astype(int)
        if len(per_layer) == expected_layers and len(set(per_layer.tolist())) == 1:
            total_generated_tokens = int(per_layer.iloc[0])

    if total_generated_tokens is not None:
        observed = float(df["selected_count"].sum())
        expected = float(total_generated_tokens * top_k * expected_layers)
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=0.5):
            inventory.update(
                status="rejected",
                reason=f"selected_count_conservation_{observed}_expected_{expected}",
            )
            return None, inventory

    inventory.update(
        status="valid_candidate",
        reason="",
        rows=len(df),
        top_k=top_k,
        total_generated_tokens=total_generated_tokens,
    )
    return (
        TaskAccounting(
            task_key=task_key,
            task_num=task_num,
            csv_path=csv_path,
            frame=df,
            top_k=top_k,
            total_generated_tokens=total_generated_tokens,
        ),
        inventory,
    )


def discover_tasks(
    accounting_root: Path,
    *,
    filename: str,
    bucket: str,
    expected_layers: int,
    expected_experts: int,
) -> tuple[list[TaskAccounting], pd.DataFrame]:
    paths = sorted(accounting_root.rglob(filename))
    if not paths:
        raise FileNotFoundError(f"No {filename} files found below {accounting_root}")

    candidates: list[TaskAccounting] = []
    inventory_rows: list[dict[str, Any]] = []
    for path in paths:
        task, row = validate_one_csv(
            path,
            bucket=bucket,
            expected_layers=expected_layers,
            expected_experts=expected_experts,
        )
        inventory_rows.append(row)
        if task is not None:
            candidates.append(task)

    # Keep the newest valid result for each task if duplicate accounting folders exist.
    grouped: dict[str, list[TaskAccounting]] = {}
    for task in candidates:
        grouped.setdefault(task.task_key, []).append(task)

    selected: list[TaskAccounting] = []
    selected_paths: set[str] = set()
    for _, group in sorted(grouped.items()):
        winner = max(group, key=lambda x: x.csv_path.stat().st_mtime)
        selected.append(winner)
        selected_paths.add(str(winner.csv_path))

    for row in inventory_rows:
        if row["status"] != "valid_candidate":
            continue
        if row["csv_path"] in selected_paths:
            row.update(selected=True, status="selected")
        else:
            row.update(
                selected=False,
                status="rejected_duplicate",
                reason="newer_valid_result_selected_for_same_task",
            )

    selected.sort(key=lambda x: (x.task_num is None, x.task_num or 0, x.task_key))
    inventory = pd.DataFrame(inventory_rows)
    if not selected:
        raise RuntimeError("No valid task accounting files remained after validation")
    return selected, inventory


def build_task_table(tasks: list[TaskAccounting]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for task in tasks:
        frame = task.frame[["layer", "expert", "selected_count", "weighted_count"]].copy()
        frame.insert(0, "task_key", task.task_key)
        frame.insert(1, "task_num", task.task_num)
        frame["source_csv"] = str(task.csv_path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_deterministic_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    output = scores.copy().sort_values(["layer", "expert"]).reset_index(drop=True)
    for score_col, prefix in [
        ("selected_count", "frequency"),
        ("weighted_count", "weighted_frequency"),
    ]:
        output[f"{prefix}_share_within_layer"] = output[score_col] / output.groupby("layer")[score_col].transform("sum").replace(0, np.nan)
        output[f"{prefix}_share_within_layer"] = output[f"{prefix}_share_within_layer"].fillna(0.0)
        rank_col = f"{prefix}_rank_within_layer"
        output[rank_col] = 0
        for layer in sorted(output["layer"].unique()):
            idx = output.index[output["layer"] == layer]
            order = output.loc[idx].sort_values(
                [score_col, "expert"], ascending=[False, True], kind="mergesort"
            ).index.tolist()
            for rank, row_index in enumerate(order, start=1):
                output.at[row_index, rank_col] = rank
        output[rank_col] = output[rank_col].astype(int)
    return output


def make_plan(
    *,
    criterion: str,
    score_col: str,
    keep_size: int,
    scores: pd.DataFrame,
    num_layers: int,
    num_experts: int,
    top_k: int,
    bucket: str,
    accounting_root: Path,
    num_tasks: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if keep_size > num_experts:
        raise ValueError(f"keep size {keep_size} exceeds {num_experts}")
    if keep_size < top_k:
        raise ValueError(f"keep size {keep_size} is smaller than router top_k={top_k}")

    keep_mask = torch.zeros((num_layers, num_experts), dtype=torch.bool)
    retained_sorted_tensor = torch.empty((num_layers, keep_size), dtype=torch.long)
    retained_ranked_tensor = torch.empty((num_layers, keep_size), dtype=torch.long)
    score_tensor = torch.zeros((num_layers, num_experts), dtype=torch.float64)
    layers: dict[str, Any] = {}

    for layer in range(num_layers):
        sub = scores[scores["layer"] == layer].sort_values(
            [score_col, "expert"], ascending=[False, True], kind="mergesort"
        )
        ranked = sub["expert"].astype(int).tolist()
        retained_ranked = ranked[:keep_size]
        retained_sorted = sorted(retained_ranked)
        retained_set = set(retained_sorted)
        pruned = [x for x in range(num_experts) if x not in retained_set]
        score_by_id = dict(zip(sub["expert"].astype(int), sub[score_col].astype(float)))
        all_scores = [float(score_by_id[i]) for i in range(num_experts)]
        total = float(sum(all_scores))
        retained_total = float(sum(score_by_id[i] for i in retained_sorted))

        keep_mask[layer, retained_sorted] = True
        retained_sorted_tensor[layer] = torch.tensor(retained_sorted)
        retained_ranked_tensor[layer] = torch.tensor(retained_ranked)
        score_tensor[layer] = torch.tensor(all_scores, dtype=torch.float64)

        layers[str(layer)] = {
            "layer": layer,
            "keep_size": keep_size,
            "retained_original_expert_ids": retained_sorted,
            "retained_original_expert_ids_by_score": retained_ranked,
            "pruned_original_expert_ids": pruned,
            "new_to_original_expert_id": retained_sorted,
            "original_to_new_expert_id": {
                str(original_id): new_id
                for new_id, original_id in enumerate(retained_sorted)
            },
            "criterion_score_by_original_expert_id": all_scores,
            "criterion_score_total": total,
            "criterion_score_retained": retained_total,
            "criterion_score_coverage": retained_total / total if total > 0 else 0.0,
        }

    plan = {
        "format_version": 1,
        "plan_type": "static_per_layer_routed_expert_pruning",
        "criterion": criterion,
        "criterion_score_column": score_col,
        "aggregation": "global_sum_over_all_selected_tasks",
        "bucket": bucket,
        "source_accounting_root": str(accounting_root),
        "num_calibration_tasks": num_tasks,
        "num_layers": num_layers,
        "original_num_routed_experts_per_layer": num_experts,
        "retained_num_routed_experts_per_layer": keep_size,
        "pruned_num_routed_experts_per_layer": num_experts - keep_size,
        "num_experts_per_token_top_k": top_k,
        "shared_expert_policy": "unchanged_not_ranked_not_pruned",
        "tie_break_rule": "score_descending_then_original_expert_id_ascending",
        "checkpoint_reindex_rule": (
            "Slice each layer using retained_original_expert_ids, which are sorted "
            "by original ID. The new expert ID is the position in that list."
        ),
        "layers": layers,
    }
    tensor_plan = {
        "keep_mask": keep_mask,
        "retained_original_expert_ids": retained_sorted_tensor,
        "retained_original_expert_ids_by_score": retained_ranked_tensor,
        "criterion_scores": score_tensor,
    }
    return plan, tensor_plan


def coverage_for_plan(
    *,
    plan_id: str,
    criterion: str,
    keep_size: int,
    plan: dict[str, Any],
    task_table: pd.DataFrame,
    global_scores: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    retained = {
        int(layer): set(info["retained_original_expert_ids"])
        for layer, info in plan["layers"].items()
    }

    layer_rows: list[dict[str, Any]] = []
    for layer, sub in global_scores.groupby("layer", sort=True):
        mask = sub["expert"].isin(retained[int(layer)])
        frequency_total = float(sub["selected_count"].sum())
        frequency_kept = float(sub.loc[mask, "selected_count"].sum())
        weighted_total = float(sub["weighted_count"].sum())
        weighted_kept = float(sub.loc[mask, "weighted_count"].sum())
        layer_rows.append({
            "plan_id": plan_id,
            "criterion": criterion,
            "keep_size": keep_size,
            "layer": int(layer),
            "frequency_coverage": frequency_kept / frequency_total if frequency_total else 0.0,
            "weighted_frequency_coverage": weighted_kept / weighted_total if weighted_total else 0.0,
        })

    task_rows: list[dict[str, Any]] = []
    for task_key, sub in task_table.groupby("task_key", sort=True):
        keep = pd.Series(
            [int(expert) in retained[int(layer)] for layer, expert in zip(sub["layer"], sub["expert"])],
            index=sub.index,
        )
        frequency_total = float(sub["selected_count"].sum())
        frequency_kept = float(sub.loc[keep, "selected_count"].sum())
        weighted_total = float(sub["weighted_count"].sum())
        weighted_kept = float(sub.loc[keep, "weighted_count"].sum())
        nums = sub["task_num"].dropna().unique().tolist()
        task_rows.append({
            "plan_id": plan_id,
            "criterion": criterion,
            "keep_size": keep_size,
            "task_key": task_key,
            "task_num": int(nums[0]) if nums else None,
            "frequency_coverage": frequency_kept / frequency_total if frequency_total else 0.0,
            "weighted_frequency_coverage": weighted_kept / weighted_total if weighted_total else 0.0,
        })

    layer_df = pd.DataFrame(layer_rows)
    task_df = pd.DataFrame(task_rows)
    global_frequency_total = float(global_scores["selected_count"].sum())
    global_weighted_total = float(global_scores["weighted_count"].sum())
    global_frequency_kept = 0.0
    global_weighted_kept = 0.0
    for layer, sub in global_scores.groupby("layer"):
        mask = sub["expert"].isin(retained[int(layer)])
        global_frequency_kept += float(sub.loc[mask, "selected_count"].sum())
        global_weighted_kept += float(sub.loc[mask, "weighted_count"].sum())

    summary = {
        "plan_id": plan_id,
        "criterion": criterion,
        "keep_size": keep_size,
        "pruned_per_layer": int(plan["original_num_routed_experts_per_layer"]) - keep_size,
        "global_frequency_coverage": global_frequency_kept / global_frequency_total if global_frequency_total else 0.0,
        "global_weighted_frequency_coverage": global_weighted_kept / global_weighted_total if global_weighted_total else 0.0,
        "mean_layer_frequency_coverage": float(layer_df["frequency_coverage"].mean()),
        "min_layer_frequency_coverage": float(layer_df["frequency_coverage"].min()),
        "mean_layer_weighted_frequency_coverage": float(layer_df["weighted_frequency_coverage"].mean()),
        "min_layer_weighted_frequency_coverage": float(layer_df["weighted_frequency_coverage"].min()),
        "mean_task_frequency_coverage": float(task_df["frequency_coverage"].mean()),
        "p10_task_frequency_coverage": float(task_df["frequency_coverage"].quantile(0.10)),
        "min_task_frequency_coverage": float(task_df["frequency_coverage"].min()),
        "mean_task_weighted_frequency_coverage": float(task_df["weighted_frequency_coverage"].mean()),
        "p10_task_weighted_frequency_coverage": float(task_df["weighted_frequency_coverage"].quantile(0.10)),
        "min_task_weighted_frequency_coverage": float(task_df["weighted_frequency_coverage"].min()),
    }
    return layer_rows, task_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate old response-token frequency accounting and create two "
            "checkpoint-ready pruning-plan families: frequency and weighted frequency."
        )
    )
    parser.add_argument("--accounting-root", default="accounting_result")
    parser.add_argument("--output-root", default="pruning_info")
    parser.add_argument("--keep-sizes", default="192,160,128,96,64,48")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument(
        "--exclude-tasks",
        default="",
        help=(
            "Comma-separated task numbers or inclusive ranges to exclude. "
            "Examples: '188' or '50-60,188'."
        ),
    )
    parser.add_argument("--aggregated-filename", default=AGGREGATED_FILENAME)
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument("--expected-experts", type=int, default=256)
    args = parser.parse_args()

    accounting_root = Path(args.accounting_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plans_root = output_root / "plans"
    plans_root.mkdir(parents=True, exist_ok=True)
    keep_sizes = parse_keep_sizes(args.keep_sizes)
    excluded_task_nums = parse_task_numbers(args.exclude_tasks)

    print(f"[info] accounting root: {accounting_root}")
    print(f"[info] output root: {output_root}")
    print(f"[info] response bucket: {args.bucket}")
    print(f"[info] keep sizes: {keep_sizes}")
    print(f"[info] explicitly excluded tasks: {sorted(excluded_task_nums)}")

    tasks, inventory = discover_tasks(
        accounting_root,
        filename=args.aggregated_filename,
        bucket=args.bucket,
        expected_layers=args.expected_layers,
        expected_experts=args.expected_experts,
    )

    if excluded_task_nums:
        kept_tasks: list[TaskAccounting] = []
        for task in tasks:
            if task.task_num is not None and task.task_num in excluded_task_nums:
                mask = (
                    inventory["selected"].eq(True)
                    & inventory["task_num"].eq(task.task_num)
                )
                inventory.loc[mask, "selected"] = False
                inventory.loc[mask, "status"] = "excluded_by_user"
                inventory.loc[mask, "reason"] = (
                    f"explicitly_excluded_task_{task.task_num:04d}"
                )
            else:
                kept_tasks.append(task)
        tasks = kept_tasks

    inventory.to_csv(output_root / "task_inventory.csv", index=False)

    if not tasks:
        raise RuntimeError(
            "No valid accounting tasks remain after applying --exclude-tasks"
        )

    top_k_values = sorted({task.top_k for task in tasks})
    if len(top_k_values) != 1:
        raise RuntimeError(f"Inconsistent top_k values: {top_k_values}")
    top_k = top_k_values[0]
    if any(k > args.expected_experts or k < top_k for k in keep_sizes):
        raise ValueError(f"Every keep size must be in [{top_k}, {args.expected_experts}]")

    task_table = build_task_table(tasks)
    task_table.to_csv(output_root / "task_layer_expert_scores.csv", index=False)
    try:
        task_table.to_parquet(output_root / "task_layer_expert_scores.parquet", index=False)
    except ImportError:
        print("[warn] pyarrow/fastparquet unavailable; skipped task parquet output")

    global_scores = (
        task_table.groupby(["layer", "expert"], as_index=False)
        .agg(
            selected_count=("selected_count", "sum"),
            weighted_count=("weighted_count", "sum"),
            tasks_with_nonzero_frequency=("selected_count", lambda x: int((x > 0).sum())),
            tasks_with_nonzero_weighted_frequency=("weighted_count", lambda x: int((x > 0).sum())),
        )
    )
    global_scores = add_deterministic_ranks(global_scores)
    global_scores.to_csv(output_root / "global_expert_scores.csv", index=False)
    try:
        global_scores.to_parquet(output_root / "global_expert_scores.parquet", index=False)
    except ImportError:
        print("[warn] pyarrow/fastparquet unavailable; skipped global-score parquet output")

    criteria = [
        ("frequency", "selected_count"),
        ("weighted_frequency", "weighted_count"),
    ]
    layer_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for criterion, score_col in criteria:
        for keep_size in keep_sizes:
            plan_id = f"{criterion}_keep_{keep_size:03d}"
            plan, tensor_plan = make_plan(
                criterion=criterion,
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
            (plans_root / f"{plan_id}.json").write_text(
                json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            torch.save(
                {
                    "format_version": 1,
                    "plan_id": plan_id,
                    "criterion": criterion,
                    "criterion_score_column": score_col,
                    "keep_size": keep_size,
                    "bucket": args.bucket,
                    "num_layers": args.expected_layers,
                    "original_num_experts": args.expected_experts,
                    "top_k": top_k,
                    **tensor_plan,
                },
                plans_root / f"{plan_id}.pt",
            )
            one_layer, one_task, one_summary = coverage_for_plan(
                plan_id=plan_id,
                criterion=criterion,
                keep_size=keep_size,
                plan=plan,
                task_table=task_table,
                global_scores=global_scores,
            )
            layer_rows.extend(one_layer)
            task_rows.extend(one_task)
            summary_rows.append(one_summary)
            print(
                f"[plan] {plan_id}: "
                f"frequency={one_summary['global_frequency_coverage']:.6f} "
                f"weighted={one_summary['global_weighted_frequency_coverage']:.6f}"
            )

    pd.DataFrame(summary_rows).to_csv(output_root / "plan_summary.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(output_root / "plan_layer_coverage.csv", index=False)
    pd.DataFrame(task_rows).to_csv(output_root / "plan_task_coverage.csv", index=False)

    metadata = {
        "mode": "old_frequency_accounting_to_pruning_info",
        "accounting_root": str(accounting_root),
        "output_root": str(output_root),
        "aggregated_filename": args.aggregated_filename,
        "bucket": args.bucket,
        "aggregation": "global_sum_over_all_selected_tasks",
        "criteria": ["frequency", "weighted_frequency"],
        "keep_sizes": keep_sizes,
        "excluded_task_nums": sorted(excluded_task_nums),
        "num_selected_tasks": len(tasks),
        "selected_task_keys": [task.task_key for task in tasks],
        "selected_task_nums": [task.task_num for task in tasks if task.task_num is not None],
        "num_layers": args.expected_layers,
        "num_experts_per_layer": args.expected_experts,
        "top_k": top_k,
        "shared_expert_policy": "unchanged_not_ranked_not_pruned",
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print(f"[done] selected tasks: {len(tasks)}")
    print(f"[done] plans created: {len(criteria) * len(keep_sizes)}")
    print(f"[done] output root: {output_root}")
    print(f"[next] inspect: {output_root / 'plan_summary.csv'}")


if __name__ == "__main__":
    main()
