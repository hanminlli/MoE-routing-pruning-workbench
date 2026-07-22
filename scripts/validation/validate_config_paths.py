#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_PATH_FIELDS = ("runner", "chat_template_path")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate repository-relative file references in checked-in JSON configs."
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures: list[str] = []

    for config_path in sorted((root / "configs").rglob("*.json")):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for field in REPOSITORY_PATH_FIELDS:
            value = payload.get(field)
            if not isinstance(value, str) or not value:
                continue
            candidate = Path(value)
            if candidate.is_absolute():
                failures.append(
                    f"{config_path.relative_to(root)}: {field} must be repository-relative: {value}"
                )
                continue
            if not (root / candidate).exists():
                failures.append(
                    f"{config_path.relative_to(root)}: missing {field} target: {value}"
                )

    if failures:
        print("CONFIG PATH CHECK: FAILED")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)
    print("CONFIG PATH CHECK: PASSED")


if __name__ == "__main__":
    main()
