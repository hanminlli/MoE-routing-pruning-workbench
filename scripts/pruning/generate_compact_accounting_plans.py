#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from routecat_moe_steering.phase1.plans import build_plan, write_plan
from routecat_moe_steering.phase1.scoring import CONTRACTS, aggregate_scores, normalize_task_number


def parse_ints(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def load_accounting(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "task_num" not in frame.columns:
        for candidate in ("task_id", "task_key", "source_run"):
            if candidate in frame.columns:
                frame["task_num"] = frame[candidate].map(normalize_task_number)
                break
    if "task_num" not in frame.columns:
        raise ValueError("accounting input must include task_num or an inferable task identifier")
    return frame


def merge_metadata(frame: pd.DataFrame, metadata_path: Path | None) -> pd.DataFrame:
    if metadata_path is None:
        return frame
    metadata = pd.read_csv(metadata_path)
    if "task_num" not in metadata.columns:
        for candidate in ("task_id", "task_key"):
            if candidate in metadata.columns:
                metadata["task_num"] = metadata[candidate].map(normalize_task_number)
                break
    if "task_num" not in metadata.columns:
        raise ValueError("task metadata must include task_num or task_id")
    metadata["task_num"] = metadata["task_num"].map(normalize_task_number)
    return frame.merge(metadata, on="task_num", how="left", validate="many_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate checkpoint-compatible pruning plans from compact per-task accounting."
    )
    parser.add_argument("--accounting", required=True)
    parser.add_argument("--experiment", required=True, choices=sorted(CONTRACTS))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--keep-sizes", default="192,128,64")
    parser.add_argument("--exclude-tasks", default="188")
    parser.add_argument("--task-metadata")
    parser.add_argument("--sector")
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument("--expected-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    accounting_path = Path(args.accounting).expanduser().resolve()
    metadata_path = Path(args.task_metadata).expanduser().resolve() if args.task_metadata else None
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    frame = merge_metadata(load_accounting(accounting_path), metadata_path)
    excluded = parse_ints(args.exclude_tasks)
    keep_sizes = parse_ints(args.keep_sizes)

    sectors: list[str | None]
    if args.experiment == "sector_weighted_mass":
        if "sector" not in frame.columns:
            raise ValueError("Experiment 4 requires --task-metadata with a sector column")
        if args.sector:
            sectors = [args.sector]
        else:
            sectors = sorted(str(x) for x in frame["sector"].dropna().unique())
        if not sectors:
            raise ValueError("no sectors found in task metadata")
    else:
        sectors = [None]

    summaries: list[dict[str, object]] = []
    for sector in sectors:
        scores = aggregate_scores(
            frame,
            experiment=args.experiment,
            excluded_tasks=excluded,
            sector=sector,
        )
        suffix = "global" if sector is None else sector.lower().replace("&", "and").replace(" ", "_")
        target = output_root / suffix
        target.mkdir(parents=True, exist_ok=True)
        scores.to_csv(target / "global_expert_scores.csv", index=False)

        calibration_tasks = int(scores["calibration_tasks"].max())
        for keep_size in keep_sizes:
            plan_id = f"{args.experiment}_{suffix}_keep_{keep_size:03d}"
            plan = build_plan(
                scores,
                experiment=args.experiment,
                keep_size=keep_size,
                expected_layers=args.expected_layers,
                original_experts=args.expected_experts,
                top_k=args.top_k,
                source=str(accounting_path),
                metadata={
                    "plan_id": plan_id,
                    "sector": sector,
                    "excluded_tasks": excluded,
                    "num_calibration_tasks": calibration_tasks,
                    "task_metadata": str(metadata_path) if metadata_path else None,
                },
            )
            plan_path = write_plan(target / "plans" / f"{plan_id}.json", plan)
            mean_coverage = sum(
                layer["criterion_score_coverage"] for layer in plan["layers"].values()
            ) / args.expected_layers
            summaries.append(
                {
                    "plan_id": plan_id,
                    "experiment": args.experiment,
                    "sector": sector,
                    "keep_size": keep_size,
                    "num_calibration_tasks": calibration_tasks,
                    "mean_layer_criterion_coverage": mean_coverage,
                    "plan_path": str(plan_path),
                }
            )
            print(f"[plan] {plan_id} -> {plan_path}")

    pd.DataFrame(summaries).to_csv(output_root / "plan_summary.csv", index=False)
    (output_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "formula": CONTRACTS[args.experiment].formula,
                "accounting": str(accounting_path),
                "task_metadata": str(metadata_path) if metadata_path else None,
                "excluded_tasks": excluded,
                "keep_sizes": keep_sizes,
                "sectors": sectors,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
