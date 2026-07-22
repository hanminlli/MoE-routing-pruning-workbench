#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

NUM_LAYERS = 40
NUM_EXPERTS = 256
TOP_K = 8
EXCLUDED_TASK = 188
EXPECTED_TASKS = set(range(220)) - {EXCLUDED_TASK}
OUTPUT_NAME = "expert_counts_prompt_and_generated_aggregated.csv"


def parse_task_id(value) -> int:
    text = str(value).strip()
    if text.lower().startswith("task_"):
        text = text[5:]
    return int(float(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", default="accounting_result")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"[fatal] input not found: {input_path}")
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"[fatal] output already exists: {output_root}; use --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    df = pd.read_csv(input_path, compression="gzip", dtype={"task_id": "string"})
    required = {"task_id", "layer", "expert", "selected_count", "weighted_count", "response_tokens", "top_k"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"[fatal] missing columns: {missing}")
    df["task_num"] = df["task_id"].map(parse_task_id)
    found = set(df["task_num"].unique())
    if EXCLUDED_TASK in found:
        raise SystemExit("[fatal] task 0188 is present")
    if found != EXPECTED_TASKS:
        raise SystemExit(f"[fatal] task-set mismatch: missing={sorted(EXPECTED_TASKS-found)}, unexpected={sorted(found-EXPECTED_TASKS)}")

    for task_num in sorted(EXPECTED_TASKS):
        task = df[df["task_num"] == task_num].copy()
        for col in ("layer", "expert", "selected_count", "weighted_count", "response_tokens", "top_k"):
            task[col] = pd.to_numeric(task[col], errors="raise")
        task["layer"] = task["layer"].astype(int)
        task["expert"] = task["expert"].astype(int)
        task["response_tokens"] = task["response_tokens"].astype(int)
        task["top_k"] = task["top_k"].astype(int)
        if len(task) != NUM_LAYERS * NUM_EXPERTS:
            raise SystemExit(f"[fatal] task {task_num:04d}: rows={len(task)}, expected=10240")
        if task.duplicated(["layer", "expert"]).any():
            raise SystemExit(f"[fatal] task {task_num:04d}: duplicate layer/expert rows")
        if set(task["layer"]) != set(range(NUM_LAYERS)):
            raise SystemExit(f"[fatal] task {task_num:04d}: invalid layers")
        for layer, sub in task.groupby("layer"):
            if set(sub["expert"]) != set(range(NUM_EXPERTS)):
                raise SystemExit(f"[fatal] task {task_num:04d}: invalid expert IDs in layer {layer}")
        if sorted(task["top_k"].unique()) != [TOP_K]:
            raise SystemExit(f"[fatal] task {task_num:04d}: invalid top_k")
        response_values = sorted(task["response_tokens"].unique())
        if len(response_values) != 1:
            raise SystemExit(f"[fatal] task {task_num:04d}: inconsistent response_tokens")
        response_tokens = int(response_values[0])
        observed = float(task["selected_count"].sum())
        expected = float(response_tokens * NUM_LAYERS * TOP_K)
        if not math.isclose(observed, expected, rel_tol=0, abs_tol=0.5):
            raise SystemExit(f"[fatal] task {task_num:04d}: selected_count={observed}, expected={expected}")

        task_dir = output_root / f"task_{task_num:04d}"
        task_dir.mkdir()
        output = pd.DataFrame({
            "bucket": "generated_output_prediction",
            "layer": task["layer"],
            "module_name": task.get("module_name", pd.Series([""] * len(task), index=task.index)),
            "expert": task["expert"],
            "tokens": task["response_tokens"],
            "top_k": task["top_k"],
            "selected_count": task["selected_count"],
            "weighted_count": task["weighted_count"],
        }).sort_values(["layer", "expert"])
        output.to_csv(task_dir / OUTPUT_NAME, index=False)
        metadata = {
            "mode": "imported_minimal_ordinary_accounting",
            "task_num": task_num,
            "num_experts": NUM_EXPERTS,
            "num_router_modules": NUM_LAYERS,
            "top_k": TOP_K,
            "total_generated_tokens": response_tokens,
            "generated_total_selected_count": int(observed),
            "expected_generated_total_selected_count": int(expected),
            "generated_matches_expected_total": True,
        }
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[ok] task {task_num:04d}: tokens={response_tokens}")

    (output_root / "manifest.json").write_text(json.dumps({
        "source": str(input_path),
        "excluded_tasks": [EXCLUDED_TASK],
        "selected_task_count": len(EXPECTED_TASKS),
        "selected_task_nums": sorted(EXPECTED_TASKS),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[done] imported tasks: {len(EXPECTED_TASKS)}")
    print(f"[done] output root: {output_root}")

if __name__ == "__main__":
    main()
