#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "pruning"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_pruning_plans import (  # noqa: E402
    AGGREGATED_FILENAME,
    discover_tasks,
    parse_task_numbers,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate accepted per-task response-token routing accounting."
    )
    parser.add_argument("--accounting-root", default="accounting_result")
    parser.add_argument("--policy", default="configs/calibration_policy.json")
    parser.add_argument("--output", default="state/accounting_validation.json")
    parser.add_argument("--minimum-selected-tasks", type=int, default=219)
    args = parser.parse_args()

    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    root = Path(args.accounting_root).expanduser().resolve()
    tasks, inventory = discover_tasks(
        root,
        filename=AGGREGATED_FILENAME,
        bucket=str(policy["bucket"]),
        expected_layers=int(policy["expected_layers"]),
        expected_experts=int(policy["expected_experts_per_layer"]),
    )

    excluded = set(int(x) for x in policy.get("exclude_tasks", []))
    selected_after_policy = [
        t for t in tasks if t.task_num is None or t.task_num not in excluded
    ]
    selected_nums = sorted(
        t.task_num for t in selected_after_policy if t.task_num is not None
    )
    missing_token_counts = sorted(
        t.task_num
        for t in selected_after_policy
        if t.task_num is not None and not t.total_generated_tokens
    )

    result = {
        "accounting_root": str(root),
        "policy": policy,
        "valid_unique_tasks_before_policy": len(tasks),
        "selected_tasks_after_policy": len(selected_after_policy),
        "selected_task_nums": selected_nums,
        "missing_positive_response_token_counts": missing_token_counts,
        "inventory_status_counts": inventory["status"].value_counts().to_dict(),
        "minimum_selected_tasks": args.minimum_selected_tasks,
        "valid": (
            len(selected_after_policy) >= args.minimum_selected_tasks
            and not missing_token_counts
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    inventory.to_csv(out.with_name("accounting_inventory.csv"), index=False)

    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(
            "Accounting validation failed. Do not build pruning plans from this input."
        )


if __name__ == "__main__":
    main()
