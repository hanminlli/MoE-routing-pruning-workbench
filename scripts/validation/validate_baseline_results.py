#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--expected-tasks", type=int, default=220)
    args = ap.parse_args()
    root = Path(args.result_root)
    final_manifest = root / "task_final_manifest.jsonl"
    attempt_manifest = root / "task_attempt_manifest.jsonl"
    if not final_manifest.is_file() or not attempt_manifest.is_file():
        raise SystemExit("[fatal] baseline manifests are missing")
    finals = [json.loads(x) for x in final_manifest.read_text().splitlines() if x.strip()]
    attempts = [json.loads(x) for x in attempt_manifest.read_text().splitlines() if x.strip()]
    if len(finals) != args.expected_tasks:
        raise SystemExit(f"[fatal] final task count={len(finals)}, expected={args.expected_tasks}")
    by_task = {}
    for row in attempts:
        by_task.setdefault(int(row["task_index"]), []).append(row)
    bad_order = []
    expected = [80, 50, 120]
    for task, rows in by_task.items():
        rows = sorted(rows, key=lambda r: int(r["attempt"]))
        budgets = [int(r["max_turns"]) for r in rows]
        if budgets != expected[:len(budgets)]:
            bad_order.append((task, budgets))
        finished_positions = [i for i, r in enumerate(rows) if r.get("status") == "finished"]
        if finished_positions and finished_positions[0] != len(rows) - 1:
            bad_order.append((task, f"trials continued after success: {budgets}"))
    if bad_order:
        raise SystemExit(f"[fatal] trial-order/early-stop violations: {bad_order[:10]}")
    success = sum(r.get("final_status") == "finished" for r in finals)
    failed = len(finals) - success
    print(f"[ok] baseline covers {len(finals)} tasks")
    print(f"[ok] fallback order 80 -> 50 -> 120 and early stopping validated")
    print(f"[info] successful tasks={success}, exhausted failures={failed}, attempts={len(attempts)}")


if __name__ == "__main__":
    main()
