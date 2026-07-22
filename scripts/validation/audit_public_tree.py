#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".json", ".jsonl", ".toml", ".yaml", ".yml",
    ".txt", ".csv", ".jinja", ".cff", ".gitignore",
}
FORBIDDEN_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "scratch_user_path": re.compile(r"/(?:home|scratch|mnt/batch/tasks)/[A-Za-z0-9._-]+/"),
}
FORBIDDEN_NAMES = {".env", "credentials.json", "secrets.json"}
FORBIDDEN_ENV_KEYS = ("AZUREML_RUN_ID", "AZUREML_WORKSPACE_ID", "AZURE_SUBSCRIPTION_ID", "AWS_SECRET_ACCESS_KEY")
LARGE_LIMIT = 10 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a repository before external publication.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures: list[str] = []

    for path in root.rglob("*"):
        if any(part in {".git", ".venv", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.name in FORBIDDEN_NAMES:
            failures.append(f"forbidden filename: {relative}")
        if path.stat().st_size > LARGE_LIMIT:
            failures.append(f"large tracked candidate ({path.stat().st_size} bytes): {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative}")
        for key in FORBIDDEN_ENV_KEYS:
            if re.search(rf"(?m)^\s*{re.escape(key)}\s*=", text):
                failures.append(f"internal_environment_key_{key}: {relative}")

    if failures:
        print("PUBLIC TREE AUDIT: FAILED")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)
    print("PUBLIC TREE AUDIT: PASSED")


if __name__ == "__main__":
    main()
