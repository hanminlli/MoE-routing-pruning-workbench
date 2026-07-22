#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local Markdown links in the repository.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures: list[str] = []

    for document in sorted(root.rglob("*.md")):
        if any(part in {".git", ".venv", "__pycache__", ".pytest_cache"} for part in document.parts):
            continue
        text = document.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                failures.append(f"{document.relative_to(root)} -> {raw_target}")

    if failures:
        print("MARKDOWN LINK CHECK: FAILED")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)
    print("MARKDOWN LINK CHECK: PASSED")


if __name__ == "__main__":
    main()
