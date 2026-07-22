from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def load_runner(path: Path):
    path = path.resolve()

    # Important:
    # scripts/baseline/run_gdpval.py imports project-local modules like:
    #   from src.stirrup_logging_client import ...
    #
    # Therefore the project root must be on sys.path before importing the runner.
    project_root = path.parents[2].resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    spec = importlib.util.spec_from_file_location("runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import runner from {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_file_block(prompt: str) -> str:
    start_marker = "Available input files:"
    end_marker = "\n\nTask instruction:"

    if start_marker not in prompt:
        raise RuntimeError("Prompt does not contain 'Available input files:'")

    start = prompt.index(start_marker)

    if end_marker not in prompt[start:]:
        raise RuntimeError("Prompt does not contain expected '\\n\\nTask instruction:' after file block")

    end = prompt.index(end_marker, start)
    return prompt[start:end]


def basenames_from_task(task: dict[str, Any]) -> list[str]:
    raw = []

    for key in ["local_reference_files", "reference_files"]:
        values = task.get(key) or []
        if isinstance(values, list):
            raw.extend(values)

    names: list[str] = []
    seen: set[str] = set()

    for x in raw:
        name = Path(str(x)).name
        if name and name not in seen:
            names.append(name)
            seen.add(name)

    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", default="artifacts/tasks")
    parser.add_argument("--runner", default="scripts/baseline/run_gdpval.py")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    runner_path = Path(args.runner)

    mod = load_runner(runner_path)

    task_paths = sorted(tasks_dir.glob("task_*/task.json"))

    if not task_paths:
        raise SystemExit(f"[fatal] no task.json files found under {tasks_dir}")

    failures: list[dict[str, Any]] = []
    checked = 0
    tasks_with_inputs = 0
    tasks_without_inputs = 0

    for task_path in task_paths:
        task = load_json(task_path)
        row_index = int(task["row_index"])

        prompt = mod.build_task_prompt(task, Path("dummy_output"))
        file_block = extract_file_block(prompt)

        expected_names = basenames_from_task(task)

        checked += 1

        if expected_names:
            tasks_with_inputs += 1
        else:
            tasks_without_inputs += 1

        problems: list[str] = []

        if expected_names and "No input files are listed for this task." in file_block:
            problems.append("task has reference files, but prompt says no input files are listed")

        for name in expected_names:
            if f"- {name}" not in file_block:
                problems.append(f"missing expected filename in prompt: {name}")

        # The prompt should expose only basenames, not absolute scratch/project paths.
        suspicious_path_markers = [
            "/scratch/",
            "/home/",
            "/mnt/",
            "/tmp/",
            "artifacts/tasks/",
        ]

        for marker in suspicious_path_markers:
            if marker in file_block:
                problems.append(f"file block exposes path marker: {marker}")

        if problems:
            failures.append(
                {
                    "row_index": row_index,
                    "task_path": str(task_path),
                    "expected_names": expected_names,
                    "file_block": file_block,
                    "problems": problems,
                }
            )

    print("checked_tasks:", checked)
    print("tasks_with_inputs:", tasks_with_inputs)
    print("tasks_without_inputs:", tasks_without_inputs)
    print("failures:", len(failures))

    if failures:
        print()
        print("FAILED TASKS:")
        for f in failures[:50]:
            print()
            print("=" * 100)
            print("row_index:", f["row_index"])
            print("task_path:", f["task_path"])
            print("expected_names:", f["expected_names"])
            print("problems:")
            for p in f["problems"]:
                print("  -", p)
            print()
            print(f["file_block"])

        if len(failures) > 50:
            print()
            print(f"... omitted {len(failures) - 50} additional failures")

        raise SystemExit(1)

    print("[ok] every task prompt file block is consistent with task.json reference files")


if __name__ == "__main__":
    main()
