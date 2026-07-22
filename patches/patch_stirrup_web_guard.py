from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import stirrup

p = Path(stirrup.__file__).resolve().parent / "tools" / "web.py"
print(f"[info] target file: {p}")
s = p.read_text(encoding="utf-8")
markers = (
    "WEB_FETCH_BUDGET_EXCEEDED",
    "REPEATED_URL_BLOCKED",
    "REPEATED_FAILED_URL_BLOCKED",
)
if all(marker in s for marker in markers):
    print("[ok] web guard is already installed")
    raise SystemExit(0)

stamp = time.strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"{p.name}.before_web_guard_{stamp}.bak")
shutil.copy2(p, backup)
print(f"[ok] backup saved to {backup}")

s = re.sub(
    r"^WEB_FETCH_TIMEOUT\s*=.*$",
    "WEB_FETCH_TIMEOUT = 20",
    s,
    count=1,
    flags=re.MULTILINE,
)

tool_start = s.find("def _get_fetch_web_page_tool(")
executor_start = s.find("    async def fetch_web_page_executor(", tool_start)
tool_end = s.find("\n    return Tool[", executor_start)
if tool_start < 0 or executor_start < 0 or tool_end < 0:
    raise SystemExit("[fatal] could not locate Stirrup's fetch-web-page tool")

prefix = s[:tool_start]
block = s[tool_start:tool_end]
suffix = s[tool_end:]
block = block.replace("stop=stop_after_attempt(3)", "stop=stop_after_attempt(1)", 1)
block = block.replace(
    "wait=wait_exponential(multiplier=1, min=1, max=10)",
    "wait=wait_exponential(multiplier=1, min=1, max=3)",
    1,
)

state_code = '''    # ROUTECAT_WEB_GUARD_STATE
    total_fetch_uses = 0
    per_url_uses: dict[str, int] = {}
    per_url_failures: dict[str, int] = {}

    MAX_TOTAL_FETCHES = 40
    MAX_SAME_URL_USES = 5
    MAX_SAME_URL_FAILURES = 3

'''
if "ROUTECAT_WEB_GUARD_STATE" not in block:
    decorator_pos = block.find("    @retry(")
    if decorator_pos < 0:
        raise SystemExit("[fatal] could not find web-fetch retry decorator")
    block = block[:decorator_pos] + state_code + block[decorator_pos:]

executor_start_local = block.find("    async def fetch_web_page_executor(")
docstring = '        """Fetch web page and extract main content as markdown using trafilatura."""\n'
doc_pos = block.find(docstring, executor_start_local)
if doc_pos < 0:
    raise SystemExit("[fatal] could not find fetch_web_page_executor docstring")
insert_pos = doc_pos + len(docstring)

guard_code = '''        # ROUTECAT_WEB_GUARD_EXECUTOR
        nonlocal total_fetch_uses

        url = params.url.strip()
        total_fetch_uses += 1
        per_url_uses[url] = per_url_uses.get(url, 0) + 1

        if total_fetch_uses > MAX_TOTAL_FETCHES:
            return ToolResult(
                content=(
                    f"<web_fetch><url>{url}</url><error>"
                    "WEB_FETCH_BUDGET_EXCEEDED: Too many web pages have already been fetched for this task. "
                    "Stop browsing now, use the information already gathered, create the deliverable, verify it, and call finish."
                    "</error></web_fetch>"
                ),
                success=False,
                metadata=WebFetchMetadata(pages_fetched=[url]),
            )

        if per_url_uses[url] > MAX_SAME_URL_USES:
            return ToolResult(
                content=(
                    f"<web_fetch><url>{url}</url><error>"
                    "REPEATED_URL_BLOCKED: This exact URL has already been requested several times. Do not request it again."
                    "</error></web_fetch>"
                ),
                success=False,
                metadata=WebFetchMetadata(pages_fetched=[url]),
            )

        if per_url_failures.get(url, 0) >= MAX_SAME_URL_FAILURES:
            return ToolResult(
                content=(
                    f"<web_fetch><url>{url}</url><error>"
                    "REPEATED_FAILED_URL_BLOCKED: This URL has already failed several times. "
                    "Use another source or proceed with the available evidence."
                    "</error></web_fetch>"
                ),
                success=False,
                metadata=WebFetchMetadata(pages_fetched=[url]),
            )

'''
if "ROUTECAT_WEB_GUARD_EXECUTOR" not in block:
    block = block[:insert_pos] + guard_code + block[insert_pos:]

executor_start_local = block.find("    async def fetch_web_page_executor(")
executor = block[executor_start_local:]
executor = executor.replace("params.url", "url")
executor = executor.replace("url = url.strip()", "url = params.url.strip()", 1)
except_line = "        except httpx.HTTPError as exc:\n"
if except_line not in executor:
    raise SystemExit("[fatal] could not find HTTP-error handler")
failure_counter = "            per_url_failures[url] = per_url_failures.get(url, 0) + 1\n"
if failure_counter not in executor:
    executor = executor.replace(except_line, except_line + failure_counter, 1)

block = block[:executor_start_local] + executor
s = prefix + block + suffix
p.write_text(s, encoding="utf-8")
result = subprocess.run([sys.executable, "-m", "py_compile", str(p)], text=True, capture_output=True)
if result.returncode != 0:
    shutil.copy2(backup, p)
    print(result.stdout)
    print(result.stderr)
    raise SystemExit("[fatal] patched file failed compilation; original restored")

checks = {
    "timeout": "WEB_FETCH_TIMEOUT = 20" in s,
    "total budget": "WEB_FETCH_BUDGET_EXCEEDED" in s,
    "same URL": "REPEATED_URL_BLOCKED" in s,
    "failed URL": "REPEATED_FAILED_URL_BLOCKED" in s,
    "failure counter": failure_counter.strip() in s,
}
for name, passed in checks.items():
    print(f"[check] {name}: {passed}")
if not all(checks.values()):
    shutil.copy2(backup, p)
    raise SystemExit("[fatal] verification failed; original restored")
print("[ok] py_compile passed")
print("[ok] version-adaptive web guard patch complete")
