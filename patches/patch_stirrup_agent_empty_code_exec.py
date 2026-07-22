from pathlib import Path
import time
import stirrup

p = Path(stirrup.__file__).resolve().parent / "core" / "agent.py"

s = p.read_text()
stamp = time.strftime("%Y%m%d_%H%M%S")
backup = p.with_suffix(p.suffix + f".bak_repeated_empty_code_exec_from_original_{stamp}")
backup.write_text(s)
print(f"[backup] {p} -> {backup}")

original_line = '                result = ToolResult(content="Tool arguments are not valid", success=False)'

if "EMPTY_CODE_EXEC_REPEATED_TOO_MANY_TIMES" in s:
    print("[ok] repeated empty code_exec escalation is already installed")
    raise SystemExit(0)

if original_line not in s:
    raise SystemExit(
        "[fatal] original invalid-tool line not found. "
        "This patch assumes you are starting from the clean original agent.py file."
    )

lines = s.splitlines()

# Find run_tool first, so we only patch the ValidationError block used for tool calls.
run_tool_start = None
for i, line in enumerate(lines):
    if line.strip().startswith("async def run_tool("):
        run_tool_start = i
        break

if run_tool_start is None:
    raise SystemExit("[fatal] could not find async def run_tool")

# Find the ValidationError block inside run_tool.
start = None
for i in range(run_tool_start, len(lines)):
    if lines[i].strip() == "except ValidationError:":
        start = i
        break

if start is None:
    raise SystemExit("[fatal] could not find except ValidationError inside run_tool")

# Replace from the line after "except ValidationError:" through "args_valid = False".
end = None
for j in range(start + 1, len(lines)):
    if lines[j].strip() == "args_valid = False":
        end = j
        break
    if lines[j].startswith("        else:"):
        raise SystemExit("[fatal] reached tool-not-found else before args_valid = False")

if end is None:
    raise SystemExit("[fatal] could not find args_valid = False after ValidationError block")

replacement = r'''                LOGGER.debug(
                    "LLMClient tried to use the tool %s but the tool arguments are not valid: %r",
                    tool_call.name,
                    tool_call.arguments,
                )

                is_empty_code_exec = (
                    tool_call.name == "code_exec"
                    and (
                        not tool_call.arguments
                        or not tool_call.arguments.strip()
                        or tool_call.arguments.strip() == "{}"
                    )
                )

                if is_empty_code_exec:
                    empty_entries = run_metadata.setdefault("_empty_code_exec_invalid", [])
                    empty_entries.append(
                        {
                            "tool_name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                    )

                    run_metadata_count = len(empty_entries)
                    self_count = getattr(self, "_empty_code_exec_invalid_count", 0) + 1
                    setattr(self, "_empty_code_exec_invalid_count", self_count)
                    empty_count = max(run_metadata_count, self_count)

                    if empty_count >= 3:
                        result = ToolResult(
                            content=(
                                "EMPTY_CODE_EXEC_REPEATED_TOO_MANY_TIMES: "
                                f"You have now made {empty_count} empty code_exec calls in this task. "
                                "Stop emitting code_exec{} immediately. "
                                "Your next tool call must be either code_exec with a non-empty cmd argument, "
                                "or finish if the deliverable files already exist. "
                                "Valid examples: {\"cmd\": \"ls -lh\"} or {\"cmd\": \"python3 create_docs.py\"}. "
                                "For this task, create the requested deliverables using a non-empty command, "
                                "verify the files with {\"cmd\": \"ls -lh\"}, and then call finish with paths."
                            ),
                            success=False,
                        )
                    else:
                        result = ToolResult(
                            content=(
                                "Tool arguments are not valid. "
                                "EMPTY_CODE_EXEC_BLOCKED: If this was code_exec, you called it with empty or malformed arguments. "
                                "A valid code_exec call must be exactly like {\"cmd\": \"ls -lh\"} or {\"cmd\": \"python3 script.py\"}. "
                                "Do not retry code_exec with {}. "
                                "If deliverable files already exist, verify them with {\"cmd\": \"ls -lh\"} and then call finish with paths."
                            ),
                            success=False,
                        )
                else:
                    result = ToolResult(
                        content=(
                            "Tool arguments are not valid. "
                            "Check the required schema for this tool and retry with valid arguments."
                        ),
                        success=False,
                    )

                args_valid = False'''.splitlines()

new_lines = lines[: start + 1] + replacement + lines[end + 1 :]
p.write_text("\n".join(new_lines) + "\n")

print("[ok] patched repeated empty code_exec escalation in agent.py")
