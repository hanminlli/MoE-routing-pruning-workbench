from pathlib import Path
import py_compile
import time

TARGET = Path("src/stirrup_logging_client.py")
PATCH_MARKER = "PATCH_EMPTY_SUMMARY_REASONING_FALLBACK_V1"

if not TARGET.exists():
    raise SystemExit(
        "[fatal] src/stirrup_logging_client.py not found. "
        "Run this script from the project root."
    )

s = TARGET.read_text(encoding="utf-8")

if PATCH_MARKER in s:
    print("[info] empty-summary reasoning fallback patch already installed")
    raise SystemExit(0)

stamp = time.strftime("%Y%m%d_%H%M%S")
backup = TARGET.with_suffix(TARGET.suffix + f".bak_empty_summary_reasoning_fallback_{stamp}")
backup.write_text(s, encoding="utf-8")
print(f"[backup] {TARGET} -> {backup}")

helper = f'''

# {PATCH_MARKER}
def _stirrup_patch_msg_content_for_summary_fallback(msg):
    if isinstance(msg, dict):
        return msg.get("content")
    return getattr(msg, "content", None)


def _stirrup_patch_response_get(resp, name):
    if isinstance(resp, dict):
        return resp.get(name)
    return getattr(resp, name, None)


def _stirrup_patch_response_set(resp, name, value):
    if isinstance(resp, dict):
        resp[name] = value
        return True
    try:
        setattr(resp, name, value)
        return True
    except Exception:
        return False


def _stirrup_patch_is_summary_request(messages):
    if not isinstance(messages, list):
        return False

    text_parts = []
    for msg in messages[-6:]:
        content = _stirrup_patch_msg_content_for_summary_fallback(msg)
        if isinstance(content, str):
            text_parts.append(content)

    joined = "\\n".join(text_parts)

    return (
        "Please create a concise summary of the conversation so far" in joined
        and "Do not use any tools" in joined
    )


def _stirrup_patch_extract_reasoning_text(resp):
    reasoning = _stirrup_patch_response_get(resp, "reasoning")

    candidates = []

    if isinstance(reasoning, dict):
        candidates.append(reasoning.get("content"))
        candidates.append(reasoning.get("text"))
    else:
        candidates.append(getattr(reasoning, "content", None))
        candidates.append(getattr(reasoning, "text", None))

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    return ""


def _stirrup_patch_recover_summary_from_reasoning(messages, resp):
    """Recover summary text when Qwen puts it in reasoning.content.

    Observed failure mode:
    - The context-summary request says "Do not use any tools."
    - Qwen returns visible content == "".
    - Qwen puts the actual summary in response.reasoning.content.
    - Qwen may also emit a tool call.
    - Stirrup continuation uses response.content, so the inserted summary is empty.

    This patch copies reasoning.content into content for summary requests only.
    It also clears tool_calls for that summary response, because a summary turn should
    be plain text and should not execute tools.
    """
    if not _stirrup_patch_is_summary_request(messages):
        return False

    content = _stirrup_patch_response_get(resp, "content")
    if isinstance(content, str) and content.strip():
        return False

    recovered = _stirrup_patch_extract_reasoning_text(resp)
    if not recovered:
        return False

    ok = _stirrup_patch_response_set(resp, "content", recovered)

    # Prevent summary turns from being interpreted as tool-using turns.
    _stirrup_patch_response_set(resp, "tool_calls", [])

    metadata = _stirrup_patch_response_get(resp, "metadata")
    if isinstance(metadata, dict):
        metadata["empty_summary_reasoning_fallback_applied"] = True
        metadata["empty_summary_reasoning_fallback_chars"] = len(recovered)
    else:
        _stirrup_patch_response_set(
            resp,
            "metadata",
            {{
                "empty_summary_reasoning_fallback_applied": True,
                "empty_summary_reasoning_fallback_chars": len(recovered),
            }},
        )

    return ok
'''

# Insert helper before LoggedStirrupClient if possible.
insert_anchor = "class LoggedStirrupClient"
if insert_anchor not in s:
    raise SystemExit("[fatal] could not find class LoggedStirrupClient in src/stirrup_logging_client.py")

s = s.replace(insert_anchor, helper + "\n" + insert_anchor, 1)

lines = s.splitlines()
generate_start = None
for i, line in enumerate(lines):
    if line.startswith("    async def generate("):
        generate_start = i
        break

if generate_start is None:
    raise SystemExit("[fatal] could not find LoggedStirrupClient.generate(...)")

# Find the first line inside generate that awaits the underlying model response.
response_line = None
for i in range(generate_start + 1, min(len(lines), generate_start + 200)):
    stripped = lines[i].strip()
    if (
        stripped.startswith("response = await ")
        and ".generate(" in stripped
    ):
        response_line = i
        break

if response_line is None:
    print("[debug] Could not find a line like: response = await <client>.generate(...)")
    print("[debug] Nearby generate() body:")
    for line in lines[generate_start:generate_start + 80]:
        print(line)
    raise SystemExit("[fatal] could not find response await line to patch")

indent = lines[response_line][: len(lines[response_line]) - len(lines[response_line].lstrip())]

patch_line = indent + "_stirrup_patch_recover_summary_from_reasoning(messages, response)"

if patch_line.strip() in s:
    print("[info] generate() already contains recovery call")
else:
    lines.insert(response_line + 1, patch_line)

new_s = "\n".join(lines) + "\n"
TARGET.write_text(new_s, encoding="utf-8")

py_compile.compile(str(TARGET), doraise=True)

print("[ok] patched empty-summary reasoning fallback")
print(f"[ok] syntax check passed: {TARGET}")
print(f"[restore] cp {backup} {TARGET}")
