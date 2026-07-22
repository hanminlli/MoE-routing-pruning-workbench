from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.core.agent import Agent

from src.stirrup_logging_client import LoggedStirrupClient, _jsonable


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def abs_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def list_output_files(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted([p for p in output_dir.rglob("*") if p.is_file()])


def visible_file_names(task: dict[str, Any]) -> list[str]:
    """
    Return only local filenames visible to the agent.

    Important:
    - The absolute source paths are still used by agent.session(input_files=...).
    - But we never expose those absolute paths in the model prompt.
    - This prevents the model from cd-ing into the task workspace and writing
      outputs outside the local execution environment.
    """
    paths = task.get("local_reference_files", [])
    names: list[str] = []

    for p in paths:
        name = Path(p).name
        if name not in names:
            names.append(name)

    return names


def build_system_prompt() -> str:
    return """You are an agent solving sandboxed file-production tasks in an execution environment.

Your objective is to create the requested deliverable files correctly and efficiently within an 80-turn budget.

Core rules:
- Follow the user task instruction exactly.
- Focus on producing the requested deliverables, not on explaining your process.
- Use the provided input files and sources required by the task.
- Do not search for benchmark solutions, reference answers, answer keys, grading rubrics, hidden files, or pre-existing solution files.
- Do not search the whole filesystem.
- Do not use /tmp as the working directory.
- Do not write outside the current working/output directory.
- Do not expose absolute scratch, project, workspace, or output-directory paths in the final result.
- Prefer simple, robust Python scripts over complicated shell pipelines.
- Prefer already installed standard/common Python libraries.
- Do not use uv.
- Never call code_exec with empty arguments. Every code_exec call must contain a non-empty shell command.
- Never repeat the exact same code_exec command after it has already returned a result.
- If a command succeeds but gives irrelevant or unhelpful output, do not run that same command again.
- If you receive EMPTY_CODE_EXEC_BLOCKED or any invalid-arguments message for code_exec, do not retry code_exec with empty arguments.
- A valid code_exec call must look like {"cmd": "ls -lh"} or {"cmd": "python3 script.py"}.
- If deliverable files already exist, verify them with a non-empty code_exec command and then call finish with paths.
- When the system says this is the last turn, do not create more files unless absolutely necessary; call finish if the deliverable exists.

80-turn execution budget:
- Turns 1-3: Identify all requested deliverables, required filenames, source files, and success criteria.
- Turns 4-15: Inspect only the most relevant local input files and required external sources.
- Turns 16-30: Complete information gathering. If a website, API, media source, form, or scraping approach fails twice, stop trying that source or strategy.
- Turns 31-50: Create the first complete version of every requested deliverable.
- Turns 51-60: Verify the deliverables and fix only concrete issues found by verification.
- Turns 61-68: Finalize formatting, filenames, and required content. Patch existing files rather than rewriting from scratch.
- Turns 69-74: Run final existence checks for every deliverable and prepare finish paths.
- Turns 75-80: Do not browse, scrape, inspect new sources, or perform broad research. If deliverables exist, call finish immediately. If something is missing, create the simplest acceptable best-effort deliverable, verify it exists, and call finish.

Completion discipline:
- A complete best-effort deliverable is better than no deliverable.
- If exact external information is unavailable, blocked, dynamic, credential-gated, or unreliable after reasonable attempts, proceed using available evidence, NA fields, clearly labelled estimates, or documented assumptions as appropriate.
- Do not let missing external data prevent creation of the requested file.
- For spreadsheet/data-extraction tasks, create the workbook even if some cells must be NA or include a data-limitation note.
- For document/report tasks, draft concise content first, then create the document file, then verify it.
- For media/design/video tasks, use local/generated/simple assets if external assets are unavailable.
- For medical, medication, legal, financial, or safety-sensitive identification tasks, do not guess unsupported identities or facts. Use unknown/NA when evidence is insufficient and include a verification note inside the deliverable if appropriate.

Web, API, and external asset rules:
- Use web access only when the task actually requires external information.
- If web research is required, gather enough credible evidence and then create the deliverable.
- Do not repeatedly retry blocked, empty, unavailable, irrelevant, or failing pages.
- Do not repeatedly call the same external API or stock-media search endpoint.
- If a stock-media/API/web search returns irrelevant results twice, stop searching that source.
- If external stock assets are unavailable or irrelevant, create the requested deliverable using local/generated/simple assets instead.
- For video/design/media tasks, it is better to create a complete deliverable with simple generated visuals than to spend many turns searching for perfect external media.
- If several source attempts fail, proceed with the best available evidence and document reasonable assumptions inside the deliverable when needed.
- Do not use web search to find benchmark-specific answers or solution artifacts.

Execution rules:
- Inspect input files only as much as needed.
- Create the requested deliverable files as soon as the needed information is available.
- If an approach fails, switch to a simpler approach instead of repeating the same failing command.
- If the same verification failure appears twice, patch the existing file deterministically instead of regenerating the whole deliverable.
- If two attempts to create a complex file fail, simplify the script and create the required file directly.
- For document, spreadsheet, PDF, image, audio, or video tasks, create the best complete deliverable possible with available local/generated assets.
- Before calling finish, verify that all requested deliverable files exist.

When the task is complete, call the finish tool.

The finish tool requires BOTH fields:
- reason: a short explanation that the requested deliverables were created.
- paths: a list of final deliverable filename(s) or simple relative path(s).

Example:
{"reason": "Created the requested deliverable.", "paths": ["relative_filename.ext"]}

Finish-path rules:
- The paths field must be a list, even if there is only one file.
- Return only relative filenames or simple relative paths inside paths.
- Do not return absolute paths.
- Do not return /scratch, /tmp, project, workspace, or output-directory paths.
"""


def build_task_prompt(task: dict[str, Any], output_dir: Path) -> str:
    # Do not expose output_dir to the model.
    _ = output_dir

    # Important bug fix:
    # Exported GDPval task.json files do not use a field named "files".
    # They use local_reference_files/reference_files.
    #
    # The previous version only read task.get("files"), so the model prompt said:
    #   Available input files:
    #   - No input files are listed for this task.
    #
    # even when Stirrup had actually copied reference files into the sandbox.
    # We expose only basenames here, never absolute scratch paths.
    files = task.get("files") or visible_file_names(task) or task.get("reference_files") or []

    file_lines: list[str] = []
    seen_file_names: set[str] = set()

    for item in files:
        if isinstance(item, dict):
            name = (
                item.get("filename")
                or item.get("name")
                or item.get("path")
                or item.get("local_name")
                or str(item)
            )
        else:
            name = str(item)

        name = Path(str(name)).name

        if name and name not in seen_file_names:
            file_lines.append(f"- {name}")
            seen_file_names.add(name)

    files_block = "\n".join(file_lines) if file_lines else "- No input files are listed for this task."

    return f"""You are solving one sandboxed task.

Available input files:
{files_block}

Task instruction:
{task["prompt"]}

Required workflow:
1. Identify the requested deliverable file or files.
2. Inspect only the relevant input files or required external sources.
3. Create all requested deliverables in the current working/output directory.
4. If the task specifies exact output filenames, use those exact filenames.
5. If the task specifies deliverable titles but not filenames, choose clear filenames with suitable extensions.
6. Verify that each requested deliverable file exists.
7. Call finish with paths as a list of relative deliverable filename(s).

Important constraints:
- Do not search for benchmark solutions, reference answers, answer keys, grading rubrics, hidden files, or pre-existing solution files.
- Do not search the whole filesystem.
- Do not use /tmp as the working directory.
- Do not write outside the current working/output directory.
- Do not return absolute paths.
- Do not call code_exec with empty arguments.
- Do not repeat the exact same code_exec command.
- If one command or search gives irrelevant output, do not rerun the same command.
- Avoid repeated browsing or repeated failed commands.
- If one approach fails, switch to a simpler approach.
- For web-research tasks, once you have enough credible information, stop browsing and create the deliverable.
- For media/design/video tasks, do not spend many turns searching for perfect stock assets. If external assets are unavailable or irrelevant, create the deliverable using local/generated/simple assets.
- For media or design tasks, use available local/generated assets if external assets are unavailable.

Finish-tool requirements:
- Call finish only after the deliverables exist.
- The finish tool arguments must use this exact shape:
  {{"reason": "Created the requested deliverable.", "paths": ["relative_filename.ext"]}}
- The paths field must be a list, even when there is only one file.
- The finish paths field must contain only relative filenames or simple relative paths.
"""


async def run_one_task(
    *,
    cfg: dict[str, Any],
    task_json_path: Path,
    runs_dir: Path,
    max_turns: int,
) -> dict[str, Any]:
    task = load_json(task_json_path)

    # Absolute paths are needed only for Stirrup to copy files into the local
    # execution environment. They are not shown to the model.
    task["local_reference_files_abs"] = [
        abs_path(p) for p in task.get("local_reference_files", [])
    ]

    row_index = int(task["row_index"])
    task_id = task["task_id"]
    task_folder_name = task["task_folder_name"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"gdpval_{task_folder_name}_{timestamp}"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    output_dir = run_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "model_calls.jsonl"
    status_path = run_dir / "status.json"
    metadata_path = run_dir / "run_metadata.json"

    # Important:
    # extra_body.return_token_ids=True asks vLLM to return:
    #   - prompt_token_ids
    #   - choices[0].token_ids
    #
    # These are needed for exact generated-token expert accounting later.
    raw_client = ChatCompletionsClient(
        model=cfg["model"],
        max_tokens=cfg.get("max_tokens_per_turn", 8192),
        base_url=cfg["vllm_base_url"],
        api_key=cfg.get("vllm_api_key", "EMPTY"),
        kwargs={
            "temperature": cfg.get("temperature", 0),
            "extra_body": {
                "return_token_ids": True,
            },
        },
    )

    client = LoggedStirrupClient(
        inner=raw_client,
        log_path=log_path,
        model=cfg["model"],
        chat_template_path=cfg.get("chat_template_path"),
        enable_thinking=cfg.get("enable_thinking"),
        request_defaults={
            "model": cfg["model"],
            "temperature": cfg.get("temperature", 0),
            "max_tokens": cfg.get("max_tokens_per_turn", 8192),
            "extra_body": {
                "return_token_ids": True,
            },
        },
        log_input_ids=False,
    )

    agent = Agent(
        client=client,
        name=f"gdpval_task_{row_index:04d}",
        max_turns=max_turns,
        system_prompt=build_system_prompt(),
        tools=None,
        context_summarization_cutoff=10.0,
        turns_remaining_warning_threshold=12,
    )

    result_record: dict[str, Any] = {
        "attempt": "optionB_prompt_v3_80turn_general_budget_prompt",
        "row_index": row_index,
        "task_id": task_id,
        "task_json_path": str(task_json_path),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "output_dir": str(output_dir),
        "status": "started",
        "finish_params": None,
        "error": None,
    }
    write_json(status_path, result_record)

    try:
        async with agent.session(
            output_dir=output_dir,
            input_files=task.get("local_reference_files_abs", []),
        ) as session_agent:
            finish_params, messages_by_turn, run_metadata = await session_agent.run(
                build_task_prompt(task, output_dir)
            )

        result_record["finish_params"] = _jsonable(finish_params)
        result_record["status"] = "finished" if finish_params is not None else "not_finished"
        result_record["output_files"] = [abs_path(p) for p in list_output_files(output_dir)]
        result_record["num_message_groups"] = len(messages_by_turn)
        result_record["run_metadata_keys"] = list(run_metadata.keys())

        write_json(
            metadata_path,
            {
                "finish_params": finish_params,
                "messages_by_turn": messages_by_turn,
                "run_metadata": run_metadata,
            },
        )

    except Exception as exc:
        result_record["status"] = "failed"
        result_record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    write_json(status_path, result_record)
    print(json.dumps(result_record, ensure_ascii=False, indent=2))
    return result_record


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run_config.json")
    parser.add_argument("--tasks-dir", default="artifacts/tasks")
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=80)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    tasks_dir = Path(args.tasks_dir)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    task_json_paths = sorted(tasks_dir.glob("task_*/task.json"))

    selected: list[Path] = []
    for p in task_json_paths:
        task = load_json(p)
        row_index = int(task["row_index"])
        if args.start <= row_index < args.end:
            selected.append(p)

    print(f"Found task json files: {len(task_json_paths)}")
    print(f"Selected tasks: {len(selected)}")
    print(f"Range: [{args.start}, {args.end})")

    summary: list[dict[str, Any]] = []

    for p in selected:
        print("\n" + "=" * 120)
        print(f"Running task: {p}")
        print("=" * 120)

        result = await run_one_task(
            cfg=cfg,
            task_json_path=p,
            runs_dir=runs_dir,
            max_turns=args.max_turns,
        )
        summary.append(result)
        write_json(runs_dir / "gdpval_run_summary.json", summary)

    print("\nFinal summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
