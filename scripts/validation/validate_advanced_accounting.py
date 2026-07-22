#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REQUIRED = [
    "metadata.json", "call_token_summary.csv",
    "expert_response_activation_aggregated.csv", "layer_conservation.csv",
    "observer_state.pt",
]


def task_num(path: Path):
    m = re.findall(r"task[_-](\d{4})", str(path))
    return int(m[-1]) if m else None


def valid(meta: dict) -> bool:
    return (
        meta.get("mode") == "response_activation_nine_statistics_v1"
        and meta.get("response_count_matches_expected_total") is True
        and meta.get("gate_sum_matches_expected_total") is True
        and meta.get("all_layer_count_checks_pass") is True
        and meta.get("all_layer_gate_checks_pass") is True
        and meta.get("all_scores_finite") is True
        and meta.get("all_scores_nonnegative") is True
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="advanced_accounting_result")
    ap.add_argument("--minimum-valid-tasks", type=int, default=1)
    args = ap.parse_args()
    root = Path(args.root)
    selected = {}
    rejected = []
    for p in root.rglob("metadata.json"):
        n = task_num(p)
        if n is None:
            continue
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            missing = [name for name in REQUIRED if not (p.parent / name).is_file()]
            if missing or not valid(meta):
                rejected.append((n, str(p), missing or ["metadata checks failed"]))
                continue
            old = selected.get(n)
            if old is None or p.stat().st_mtime > old.stat().st_mtime:
                selected[n] = p
        except Exception as exc:
            rejected.append((n, str(p), [str(exc)]))
    if len(selected) < args.minimum_valid_tasks:
        raise SystemExit(
            f"[fatal] valid advanced-accounting tasks={len(selected)}, minimum={args.minimum_valid_tasks}"
        )
    print(f"[ok] valid unique advanced-accounting tasks={len(selected)}")
    print(f"[info] rejected/duplicate-invalid candidates={len(rejected)}")
    print("[warning] Experiments 1–3 do not consume advanced accounting.")


if __name__ == "__main__":
    main()
