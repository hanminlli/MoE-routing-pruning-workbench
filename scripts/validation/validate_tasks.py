#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ANSWER_MARKERS = {
    "deliverable_files", "deliverable_text", "local_deliverable_files",
    "deliverable_file_paths", "deliverable_file_urls", "hidden_reference_files",
    "rubric_pretty", "rubric_json", "gold", "gold_answer", "answer",
    "solution", "reference_answer", "ground_truth", "raw_row",
}


def walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from walk_keys(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="artifacts/tasks")
    ap.add_argument("--expected", type=int, default=220)
    args = ap.parse_args()
    root = Path(args.tasks_dir)
    task_files = sorted(root.glob("task_*/task.json"))
    problems = []
    total_refs = 0
    for p in task_files:
        try:
            task = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append(f"{p}: invalid JSON: {exc}")
            continue
        bad_keys = sorted(set(walk_keys(task)) & ANSWER_MARKERS)
        if bad_keys:
            problems.append(f"{p}: answer-side keys remain: {bad_keys}")
        refs = task.get("local_reference_files") or []
        if not isinstance(refs, list):
            problems.append(f"{p}: local_reference_files is not a list")
            continue
        for ref in refs:
            total_refs += 1
            rp = Path(ref)
            if not rp.is_file():
                problems.append(f"{p}: missing reference file: {ref}")
            if "/inputs/" not in rp.as_posix():
                problems.append(f"{p}: reference is not self-contained under inputs/: {ref}")
    if len(task_files) != args.expected:
        problems.append(f"task count={len(task_files)}, expected={args.expected}")
    if problems:
        print("[fatal] task validation failed; first problems:")
        for x in problems[:50]:
            print(" -", x)
        raise SystemExit(1)
    print(f"[ok] validated {len(task_files)} sanitized self-contained tasks")
    print(f"[ok] validated {total_refs} local reference-file entries")


if __name__ == "__main__":
    main()
