#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ANSWER_SIDE_KEYS = {
    "deliverable_files",
    "deliverable_text",
    "local_deliverable_files",
    "deliverable_file_paths",
    "deliverable_file_urls",
    "deliverable_file_hf_uris",
    "hidden_reference_files",
    "rubric_pretty",
    "rubric_json",
    "raw_row",
    "gold",
    "gold_answer",
    "answer",
    "solution",
    "reference_answer",
    "ground_truth",
}


def short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:8]


def sanitize_keys(obj: Any, removed: dict[str, Any]) -> Any:
    """Remove answer-side keys recursively, but do not remove ordinary text values."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ANSWER_SIDE_KEYS:
                removed[k] = v
                continue
            out[k] = sanitize_keys(v, removed)
        return out

    if isinstance(obj, list):
        return [sanitize_keys(x, removed) for x in obj]

    return obj


def ref_to_path(ref: Any) -> str | None:
    if isinstance(ref, str):
        return ref

    if isinstance(ref, dict):
        for k in ["local_path", "path", "filename", "file_path", "download_path"]:
            v = ref.get(k)
            if isinstance(v, str) and v:
                return v

    return None


def resolve_existing_path(path_text: str, task_dir: Path, project_dir: Path) -> Path | None:
    p = Path(path_text)

    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            project_dir / p,
            task_dir / p,
            p,
        ])

    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return c.resolve()
        except OSError:
            pass

    return None


def unique_dest(inputs_dir: Path, src: Path, used_names: set[str]) -> Path:
    name = src.name

    if name not in used_names:
        used_names.add(name)
        return inputs_dir / name

    stem = src.stem
    suffix = src.suffix
    h = short_hash(str(src.resolve()))
    name2 = f"{stem}__{h}{suffix}"

    if name2 not in used_names:
        used_names.add(name2)
        return inputs_dir / name2

    i = 2
    while True:
        name3 = f"{stem}__{h}_{i}{suffix}"
        if name3 not in used_names:
            used_names.add(name3)
            return inputs_dir / name3
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", default="artifacts/tasks")
    parser.add_argument("--backup-dir", default="../outputs/gdpval_task_gold_backup/self_contained_export")
    parser.add_argument("--project-dir", default=".")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    backup_dir = Path(args.backup_dir)
    project_dir = Path(args.project_dir).resolve()

    if not tasks_dir.exists():
        raise SystemExit(f"[fatal] tasks dir does not exist: {tasks_dir}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = backup_dir / "self_contained_manifest.jsonl"

    num_tasks = 0
    num_modified = 0
    total_refs = 0
    total_copied_refs = 0
    missing_refs: list[dict[str, Any]] = []
    removed_key_counts: dict[str, int] = {}

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for task_json in sorted(tasks_dir.glob("task_*/task.json")):
            num_tasks += 1
            task_dir = task_json.parent
            inputs_dir = task_dir / "inputs"
            inputs_dir.mkdir(exist_ok=True)

            original = json.loads(task_json.read_text(encoding="utf-8"))

            full_backup = backup_dir / f"{task_dir.name}.full_task.json"
            full_backup.write_text(
                json.dumps(original, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            removed: dict[str, Any] = {}
            sanitized = sanitize_keys(original, removed)

            for k in removed:
                removed_key_counts[k] = removed_key_counts.get(k, 0) + 1

            old_local_refs = original.get("local_reference_files") or []
            if not isinstance(old_local_refs, list):
                old_local_refs = []

            new_local_refs: list[str] = []
            visible_files: list[str] = []
            used_names: set[str] = set()

            for ref in old_local_refs:
                ref_text = ref_to_path(ref)
                if not ref_text:
                    missing_refs.append({
                        "task": task_dir.name,
                        "task_json": str(task_json),
                        "ref": ref,
                        "reason": "could_not_parse_reference_path",
                    })
                    continue

                src = resolve_existing_path(ref_text, task_dir, project_dir)
                if src is None:
                    missing_refs.append({
                        "task": task_dir.name,
                        "task_json": str(task_json),
                        "ref": ref_text,
                        "reason": "reference_file_not_found",
                    })
                    continue

                total_refs += 1

                dst = unique_dest(inputs_dir, src, used_names)

                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)

                total_copied_refs += 1
                new_local_refs.append(str(dst.resolve()))
                visible_files.append(dst.name)

            sanitized["local_reference_files"] = new_local_refs

            # This helps the prompt builder list the visible input filenames.
            # It should contain reference/input filenames only, not answer filenames.
            sanitized["files"] = visible_files

            removed_backup = backup_dir / f"{task_dir.name}.removed_answer_side_fields.json"
            removed_backup.write_text(
                json.dumps(removed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            task_json.write_text(
                json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            num_modified += 1

            manifest.write(
                json.dumps(
                    {
                        "task": task_dir.name,
                        "task_json": str(task_json),
                        "full_backup": str(full_backup),
                        "removed_backup": str(removed_backup),
                        "num_old_local_reference_files": len(old_local_refs),
                        "num_new_local_reference_files": len(new_local_refs),
                        "visible_files": visible_files,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("make_exported_tasks_self_contained")
    print("tasks_dir:", tasks_dir)
    print("backup_dir:", backup_dir)
    print("manifest:", manifest_path)
    print("num_tasks_seen:", num_tasks)
    print("num_tasks_modified:", num_modified)
    print("total_reference_files_seen:", total_refs)
    print("total_reference_files_copied:", total_copied_refs)
    print("removed_answer_side_key_counts:")
    for k, v in sorted(removed_key_counts.items()):
        print(f"  {k}: {v}")

    print("missing_reference_files:", len(missing_refs))
    if missing_refs:
        missing_path = backup_dir / "missing_reference_files.json"
        missing_path.write_text(
            json.dumps(missing_refs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("[fatal] missing reference file details:", missing_path)
        raise SystemExit(1)

    print("[ok] exported tasks are now self-contained and agent-facing task.json files are sanitized")


if __name__ == "__main__":
    main()
