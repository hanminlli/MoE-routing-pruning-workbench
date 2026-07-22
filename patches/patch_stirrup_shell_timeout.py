#!/usr/bin/env python3
from pathlib import Path
import py_compile
import re
import shutil
import time
import stirrup

root = Path(stirrup.__file__).resolve().parent
target = root / "tools" / "code_backends" / "base.py"
if not target.is_file():
    raise SystemExit(f"[fatal] cannot find expected Stirrup file: {target}")

text = target.read_text(encoding="utf-8")
if re.search(r"^\s*SHELL_TIMEOUT\s*(?::\s*int\s*)?=\s*600(?:\.0)?\b", text, re.MULTILINE):
    print(f"[ok] SHELL_TIMEOUT is already 600 seconds: {target}")
    raise SystemExit(0)

stamp = time.strftime("%Y%m%d_%H%M%S")
backup = target.with_suffix(target.suffix + f".bak_shell_timeout_to_600_{stamp}")
shutil.copy2(target, backup)

new = re.sub(
    r"(^\s*SHELL_TIMEOUT\s*(?::\s*int\s*)?=\s*)60\s*\*\s*5\b",
    r"\g<1>60 * 10",
    text,
    count=1,
    flags=re.MULTILINE,
)
if new == text:
    new = re.sub(
        r"(^\s*SHELL_TIMEOUT\s*(?::\s*int\s*)?=\s*)300(?:\.0)?\b",
        r"\g<1>600",
        text,
        count=1,
        flags=re.MULTILINE,
    )
if new == text:
    raise SystemExit("[fatal] could not locate the expected SHELL_TIMEOUT assignment")

target.write_text(new, encoding="utf-8")
py_compile.compile(str(target), doraise=True)
print(f"[ok] patched SHELL_TIMEOUT to 600 seconds: {target}")
print(f"[backup] {backup}")
