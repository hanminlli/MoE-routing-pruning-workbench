#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from routecat_moe_steering.phase1.scoring import normalize_task_number


def slugify(value: str) -> str:
    return value.lower().replace("&", "and").replace("/", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic task lists for sector-local evaluation.")
    parser.add_argument("--task-metadata", required=True)
    parser.add_argument("--output-root", default="pruning_info/experiment_4_sector_weighted_mass/task_selection")
    parser.add_argument("--sector")
    parser.add_argument("--largest-sector", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.task_metadata)
    if "task_num" not in frame.columns:
        if "task_id" not in frame.columns:
            raise ValueError("metadata requires task_num or task_id")
        frame["task_num"] = frame["task_id"].map(normalize_task_number)
    else:
        frame["task_num"] = frame["task_num"].map(normalize_task_number)
    if "sector" not in frame.columns:
        raise ValueError("metadata requires a sector column")

    counts = (
        frame.groupby("sector", as_index=False)["task_num"]
        .nunique()
        .rename(columns={"task_num": "num_tasks"})
        .sort_values(["num_tasks", "sector"], ascending=[False, True], kind="mergesort")
    )
    if args.largest_sector:
        sectors = [str(counts.iloc[0]["sector"])]
    elif args.sector:
        sectors = [args.sector]
    else:
        sectors = sorted(str(x) for x in frame["sector"].dropna().unique())

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output_root / "sector_counts.csv", index=False)

    for sector in sectors:
        tasks = sorted(frame.loc[frame["sector"].astype(str) == sector, "task_num"].unique().tolist())
        payload = {
            "sector": sector,
            "sector_slug": slugify(sector),
            "num_tasks": len(tasks),
            "task_indices": tasks,
            "task_ids": [f"{x:04d}" for x in tasks],
        }
        path = output_root / f"{slugify(sector)}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[tasks] {sector}: {len(tasks)} -> {path}")


if __name__ == "__main__":
    main()
