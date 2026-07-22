#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def value_after_flag(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    i = args.index(flag)
    if i + 1 >= len(args):
        return None
    return args[i + 1]


def main() -> None:
    here = Path(__file__).resolve().parent
    project_dir = here.parents[1]

    raw_exporter = here / "export_gdpval_tasks_raw.py"
    postprocessor = here / "make_tasks_self_contained.py"

    if not raw_exporter.exists():
        raise SystemExit(f"[fatal] missing raw exporter: {raw_exporter}")

    if not postprocessor.exists():
        raise SystemExit(f"[fatal] missing postprocessor: {postprocessor}")

    args = sys.argv[1:]

    # The current exporter normally writes to artifacts/tasks.
    # If a future exporter supports --tasks-dir, respect it.
    tasks_dir = (
        value_after_flag(args, "--tasks-dir")
        or value_after_flag(args, "--out-dir")
        or "artifacts/tasks"
    )

    backup_dir = os.environ.get(
        "GDPVAL_GOLD_BACKUP_DIR",
        "../outputs/gdpval_task_gold_backup/self_contained_export",
    )

    print("[wrapper] running raw GDPval exporter")
    subprocess.run(
        [sys.executable, str(raw_exporter), *args],
        cwd=str(project_dir),
        check=True,
    )

    print("[wrapper] making exported tasks self-contained and removing answer-side fields")
    subprocess.run(
        [
            sys.executable,
            str(postprocessor),
            "--tasks-dir",
            tasks_dir,
            "--backup-dir",
            backup_dir,
            "--project-dir",
            ".",
        ],
        cwd=str(project_dir),
        check=True,
    )

    print("[wrapper] export complete: self-contained sanitized tasks ready for agent")


if __name__ == "__main__":
    main()
