#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

FILENAME = "expert_counts_prompt_and_generated_aggregated.csv"
BUCKET = "generated_output_prediction"


def infer_task_num(path: Path) -> int:
    matches = re.findall(r"task[_-](\d{4})", str(path))
    if not matches:
        raise ValueError(f"cannot infer task number from {path}")
    return int(matches[-1])


def metadata_for(csv_path: Path) -> dict[str, object]:
    path = csv_path.parent / "metadata.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def validate_candidate(csv_path: Path) -> tuple[pd.DataFrame, int, int]:
    metadata = metadata_for(csv_path)
    if metadata.get("generated_matches_expected_total") is False:
        raise ValueError("metadata reports failed generated-token conservation")

    frame = pd.read_csv(csv_path)
    required = {"bucket", "layer", "expert", "selected_count", "weighted_count"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    frame = frame[frame["bucket"].astype(str) == BUCKET].copy()
    if frame.empty:
        raise ValueError(f"bucket {BUCKET!r} is absent")

    for column in ("layer", "expert", "selected_count", "weighted_count"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["layer"] = frame["layer"].astype(int)
    frame["expert"] = frame["expert"].astype(int)
    frame = (
        frame.groupby(["layer", "expert"], as_index=False)
        .agg(
            module_name=("module_name", "first") if "module_name" in frame.columns else ("layer", lambda x: ""),
            selected_count=("selected_count", "sum"),
            weighted_count=("weighted_count", "sum"),
            tokens=("tokens", "max") if "tokens" in frame.columns else ("selected_count", lambda x: 0),
            top_k=("top_k", "max") if "top_k" in frame.columns else ("selected_count", lambda x: 8),
        )
        .sort_values(["layer", "expert"])
        .reset_index(drop=True)
    )

    expected_pairs = {(layer, expert) for layer in range(40) for expert in range(256)}
    observed_pairs = set(zip(frame["layer"], frame["expert"]))
    if observed_pairs != expected_pairs:
        raise ValueError(
            f"expected 40x256 layer/expert grid; missing={len(expected_pairs-observed_pairs)}, "
            f"extra={len(observed_pairs-expected_pairs)}"
        )

    response_tokens = int(metadata.get("total_generated_tokens", 0) or 0)
    if not response_tokens and "tokens" in frame.columns:
        values = sorted(set(int(x) for x in frame["tokens"] if int(x) > 0))
        if len(values) == 1:
            response_tokens = values[0]
    if response_tokens <= 0:
        raise ValueError("no positive response-token count")

    top_k = int(metadata.get("top_k", 0) or 0)
    if not top_k:
        values = sorted(set(int(x) for x in frame["top_k"] if int(x) > 0))
        if len(values) == 1:
            top_k = values[0]
    if top_k <= 0:
        top_k = 8

    observed_total = float(frame["selected_count"].sum())
    expected_total = float(response_tokens * 40 * top_k)
    if abs(observed_total - expected_total) > 0.5:
        raise ValueError(
            f"selected-count conservation failed: observed={observed_total}, expected={expected_total}"
        )
    return frame, response_tokens, top_k


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine validated per-task ordinary routing accounting into one compact CSV.gz."
    )
    parser.add_argument("--accounting-root", required=True)
    parser.add_argument(
        "--output",
        default="accounting_result/ordinary_response_routing_by_task.csv.gz",
    )
    parser.add_argument("--exclude-tasks", default="188")
    parser.add_argument("--manifest", default="accounting_result/ordinary_compaction_manifest.json")
    args = parser.parse_args()

    root = Path(args.accounting_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    excluded = {int(x.strip()) for x in args.exclude_tasks.split(",") if x.strip()}

    candidates = sorted(root.rglob(FILENAME))
    if not candidates:
        raise SystemExit(f"No {FILENAME} files found below {root}")

    valid: dict[int, list[tuple[float, Path, pd.DataFrame, int, int]]] = {}
    rejected: list[dict[str, str]] = []
    for csv_path in candidates:
        try:
            task_num = infer_task_num(csv_path)
            if task_num in excluded:
                continue
            frame, response_tokens, top_k = validate_candidate(csv_path)
            valid.setdefault(task_num, []).append(
                (csv_path.stat().st_mtime, csv_path, frame, response_tokens, top_k)
            )
        except Exception as exc:
            rejected.append({"path": str(csv_path), "reason": str(exc)})

    selected = {
        task_num: max(group, key=lambda item: item[0])
        for task_num, group in valid.items()
    }
    if not selected:
        raise SystemExit("No valid accounting candidates remain")

    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    selected_manifest: list[dict[str, object]] = []
    for task_num, (_mtime, csv_path, frame, response_tokens, top_k) in sorted(selected.items()):
        sub = frame.copy()
        sub.insert(0, "task_num", task_num)
        sub.insert(1, "task_id", f"{task_num:04d}")
        sub.insert(2, "source_run", csv_path.parent.name)
        sub["response_tokens"] = response_tokens
        sub["top_k"] = top_k
        rows.append(
            sub[
                [
                    "task_num",
                    "task_id",
                    "source_run",
                    "layer",
                    "module_name",
                    "expert",
                    "selected_count",
                    "weighted_count",
                    "response_tokens",
                    "top_k",
                ]
            ]
        )
        selected_manifest.append(
            {
                "task_num": task_num,
                "source": str(csv_path),
                "response_tokens": response_tokens,
                "top_k": top_k,
            }
        )

    combined = pd.concat(rows, ignore_index=True)
    with gzip.open(output, "wt", encoding="utf-8", newline="") as handle:
        combined.to_csv(handle, index=False)

    manifest = {
        "source_accounting_root": str(root),
        "output": str(output),
        "excluded_tasks": sorted(excluded),
        "selected_task_count": len(selected),
        "selected_task_nums": sorted(selected),
        "rows": int(len(combined)),
        "expected_rows": int(len(selected) * 40 * 256),
        "duplicate_candidates": {
            f"{task_num:04d}": len(group)
            for task_num, group in valid.items()
            if len(group) > 1
        },
        "rejected_candidates": rejected,
        "selected": selected_manifest,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if len(combined) != manifest["expected_rows"]:
        raise SystemExit("combined row-count validation failed")
    print(f"[done] compact accounting: {output}")
    print(f"[done] selected tasks: {len(selected)}")
    print(f"[done] rows: {len(combined)}")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
