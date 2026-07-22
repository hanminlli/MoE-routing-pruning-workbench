#!/usr/bin/env python3
from pathlib import Path
import importlib.metadata
import py_compile
import stirrup

root = Path(stirrup.__file__).resolve().parent
checks = {
    "chat token IDs": (
        root / "clients/chat_completions_client.py",
        "_last_prompt_token_ids",
    ),
    "chat truncation fallback": (
        root / "clients/chat_completions_client.py",
        "MODEL_OUTPUT_TRUNCATED_BY_MAX_TOKENS",
    ),
    "web total-fetch guard": (
        root / "tools/web.py",
        "WEB_FETCH_BUDGET_EXCEEDED",
    ),
    "web repeated-URL guard": (
        root / "tools/web.py",
        "REPEATED_URL_BLOCKED",
    ),
    "web repeated-failure guard": (
        root / "tools/web.py",
        "REPEATED_FAILED_URL_BLOCKED",
    ),
    "empty code_exec guard": (
        root / "core/agent.py",
        "EMPTY_CODE_EXEC_REPEATED_TOO_MANY_TIMES",
    ),
}

all_ok = True
print("stirrup version:", importlib.metadata.version("stirrup"))
print("stirrup root:", root)
for name, (path, marker) in checks.items():
    present = path.is_file() and marker in path.read_text(encoding="utf-8")
    print(f"{name:30s}: {'OK' if present else 'MISSING'}")
    all_ok &= present

from stirrup.tools.code_backends import base
shell_ok = base.SHELL_TIMEOUT == 600
print(f"{'shell timeout runtime':30s}: {'OK' if shell_ok else 'MISSING'} ({base.SHELL_TIMEOUT})")
all_ok &= shell_ok

for path in sorted({path for path, _ in checks.values()} | {root / "tools/code_backends/base.py"}):
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        print(f"compile failed: {path}: {exc}")
        all_ok = False

print("ALL STIRRUP PATCHES VERIFIED:", all_ok)
if not all_ok:
    raise SystemExit(1)
