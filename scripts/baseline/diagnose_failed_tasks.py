#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PATTERNS = [
    (
        "context_length_exceeded_http400",
        [
            "maximum context length",
            "requested",
            "output tokens",
            "262144",
            "BadRequestError",
            "HTTP 400",
        ],
        "Do not change max_tokens_per_turn: RouteCat pins it at 32768. Inspect truncation, tool-call, and task-timeout evidence instead.",
    ),
    (
        "max_turns_reached",
        [
            "Maximum number of turns reached",
            "agent was not able to finish",
            "max turns",
        ],
        "Increase --max-turns for this task, and/or strengthen prompt to create deliverable once enough information is available.",
    ),
    (
        "task_timeout",
        [
            "Command exited with non-zero status 124",
            "timeout",
            "TASK_TIMEOUT",
            "timed out",
        ],
        "Increase task timeout only if the task was making progress; otherwise patch prompt/tool behavior.",
    ),
    (
        "code_exec_timeout",
        [
            "<error_kind>timeout</error_kind>",
            "Command timed out after",
        ],
        "Likely slow tool command. For media/heavy file tasks, increase code_exec timeout or add prompt guard against inefficient loops.",
    ),
    (
        "context_overflow_finish_reason_length",
        [
            "ContextOverflowError",
            "finish reason: length",
            "Reduce agent.max_tokens",
        ],
        "Reduce max_tokens_per_turn or avoid huge tool-output accumulation.",
    ),
    (
        "empty_or_invalid_code_exec",
        [
            "EMPTY_CODE_EXEC_BLOCKED",
            "invalid-arguments",
            "empty arguments",
        ],
        "Prompt/tool guard issue. Should already be mostly patched.",
    ),
    (
        "infrastructure_missing_runner_path",
        [
            "can't open file",
            "scripts/baseline/run_gdpval.py",
            "no_run_dir",
        ],
        "Infrastructure path/working-directory bug. Do not retry model; fix driver.",
    ),
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def classify_text(text: str) -> tuple[str, str, str]:
    lower = text.lower()

    for cause, needles, fix in PATTERNS:
        matched = []
        for n in needles:
            if n.lower() in lower:
                matched.append(n)

        if matched:
            # Extract a compact evidence line around the first matched phrase.
            first = matched[0].lower()
            idx = lower.find(first)
            start = max(0, idx - 500)
            end = min(len(text), idx + 1500)
            evidence = text[start:end].replace("\n", "\\n")
            return cause, evidence[:3000], fix

    return "unknown", text[-3000:].replace("\n", "\\n"), "Inspect the per-attempt logs manually."


def parse_failed_tasks(out_dir: Path) -> list[str]:
    failed_tsv = out_dir / "failed_tasks.tsv"

    if failed_tsv.exists():
        tasks = []
        for line in failed_tsv.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                tasks.append(parts[1].zfill(4))
        return tasks

    # Fallback from known failed tasks.
    return ["0051", "0120", "0129", "0155", "0188"]


def parse_attempt_manifest(out_dir: Path) -> dict[str, list[dict]]:
    p = out_dir / "task_attempt_manifest.jsonl"
    by_task: dict[str, list[dict]] = {}

    if not p.exists():
        return by_task

    for line in p.read_text(encoding="utf-8", errors="replace").split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue

        task_num = str(
            row.get("task_num")
            or row.get("task_id_num")
            or row.get("task_index")
            or ""
        ).zfill(4)

        # Prefer explicit task_num if present; otherwise try to infer from run_dir/log_path.
        blob = json.dumps(row)
        m = re.search(r"gdpval_task_(\d{4})__", blob)
        if m:
            task_num = m.group(1)

        if task_num:
            by_task.setdefault(task_num, []).append(row)

    return by_task


def find_task_logs(out_dir: Path, task_num: str) -> list[Path]:
    logs = []

    task_logs = out_dir / "task_logs"
    if task_logs.exists():
        logs.extend(sorted(task_logs.glob(f"task_{task_num}_attempt_*.log")))

    # Some drivers may store logs elsewhere.
    logs.extend(sorted(out_dir.glob(f"**/*{task_num}*.log")))

    # Deduplicate.
    seen = set()
    uniq = []
    for p in logs:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)

    return uniq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    failed_tasks = parse_failed_tasks(out_dir)
    attempt_manifest = parse_attempt_manifest(out_dir)

    report_rows = []
    evidence_dir = out_dir / "failed_task_diagnosis_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    for task_num in failed_tasks:
        logs = find_task_logs(out_dir, task_num)
        attempts = attempt_manifest.get(task_num, [])

        combined = []
        for p in logs:
            combined.append(f"\n\n===== LOG FILE: {p} =====\n")
            combined.append(read_text(p))

        if attempts:
            combined.append("\n\n===== ATTEMPT MANIFEST ROWS =====\n")
            for row in attempts:
                combined.append(json.dumps(row, ensure_ascii=False, indent=2))
                combined.append("\n")

        combined_text = "".join(combined)

        cause, evidence, fix = classify_text(combined_text)

        evidence_path = evidence_dir / f"task_{task_num}_evidence.txt"
        evidence_path.write_text(combined_text[-20000:], encoding="utf-8", errors="replace")

        report_rows.append({
            "task_num": task_num,
            "primary_cause": cause,
            "num_task_logs_found": len(logs),
            "num_attempt_manifest_rows": len(attempts),
            "evidence_file": str(evidence_path),
            "evidence_excerpt": evidence,
            "recommended_fix": fix,
        })

    out_tsv = out_dir / "failed_task_diagnosis.tsv"
    out_json = out_dir / "failed_task_diagnosis.json"

    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "task_num",
            "primary_cause",
            "num_task_logs_found",
            "num_attempt_manifest_rows",
            "evidence_file",
            "recommended_fix",
            "evidence_excerpt",
        ]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in report_rows:
            w.writerow(row)

    out_json.write_text(json.dumps(report_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 100)
    print("FAILED TASK DIAGNOSIS")
    print("=" * 100)
    print("out_dir:", out_dir)
    print("failed_tasks:", ", ".join(failed_tasks))
    print("wrote:", out_tsv)
    print("wrote:", out_json)
    print()

    for r in report_rows:
        print("-" * 100)
        print("task:", r["task_num"])
        print("cause:", r["primary_cause"])
        print("logs_found:", r["num_task_logs_found"])
        print("attempt_rows:", r["num_attempt_manifest_rows"])
        print("fix:", r["recommended_fix"])
        print("evidence_file:", r["evidence_file"])
        print("evidence_excerpt:")
        print(r["evidence_excerpt"][:1200])
        print()

    unknown = [r for r in report_rows if r["primary_cause"] == "unknown"]
    if unknown:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
