from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    if hasattr(obj, "model_dump"):
        try:
            return _jsonable(obj.model_dump())
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            return _jsonable(obj.dict())
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return _jsonable(vars(obj))
        except Exception:
            pass

    return repr(obj)


def _schema_from_param_obj(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None

    if hasattr(obj, "model_json_schema"):
        try:
            schema = obj.model_json_schema()
            if isinstance(schema, dict):
                return _jsonable(schema)
        except Exception:
            pass

    if hasattr(obj, "schema"):
        try:
            schema = obj.schema()
            if isinstance(schema, dict):
                return _jsonable(schema)
        except Exception:
            pass

    return None


def _parse_json_if_possible(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _minimal_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


def _known_tool_schema(name: str) -> dict[str, Any]:
    if name == "finish":
        return {
            "type": "object",
            "title": "FinishParams",
            "properties": {
                "reason": {
                    "type": "string",
                    "title": "Reason",
                    "description": "Reason for finishing.",
                },
                "paths": {
                    "type": "array",
                    "title": "Paths",
                    "description": "List of file paths created or modified. Do not include directories, only files.",
                    "items": {"type": "string"},
                },
            },
            "required": ["reason", "paths"],
        }

    if name == "code_exec":
        return {
            "type": "object",
            "title": "CodeExecParams",
            "properties": {
                "cmd": {
                    "type": "string",
                    "title": "Cmd",
                    "description": (
                        "Shell command to execute (bash syntax). IMPORTANT: Use only relative paths. "
                        "Do not use absolute paths (starting with / or ~) or reference directories "
                        "outside the working directory."
                    ),
                },
            },
            "required": ["cmd"],
        }

    if name == "fetch_web_page":
        return {
            "type": "object",
            "title": "FetchWebPageParams",
            "properties": {
                "url": {
                    "type": "string",
                    "title": "Url",
                    "description": "Full HTTP or HTTPS URL of the web page to fetch and extract.",
                },
            },
            "required": ["url"],
        }

    return _minimal_schema()


def normalize_logged_tools(tools: Any) -> list[dict[str, Any]]:
    if not tools:
        return []

    if isinstance(tools, list):
        if all(isinstance(t, dict) and "type" in t and "function" in t for t in tools):
            return _jsonable(tools)

    if isinstance(tools, dict):
        iterable = list(tools.items())
    elif isinstance(tools, list):
        iterable = []
        for i, tool in enumerate(tools):
            if isinstance(tool, dict):
                fallback_name = tool.get("name") or tool.get("tool_name") or f"tool_{i}"
            else:
                fallback_name = getattr(tool, "name", None) or getattr(tool, "__name__", None) or f"tool_{i}"
            iterable.append((fallback_name, tool))
    else:
        return []

    out: list[dict[str, Any]] = []

    for fallback_name, spec in iterable:
        name = str(fallback_name)
        description = ""
        parameters: Any = None

        if isinstance(spec, dict):
            name = str(spec.get("name") or spec.get("tool_name") or fallback_name)
            description = str(spec.get("description") or spec.get("desc") or "")
            parameters = spec.get("parameters")
        else:
            name = str(
                getattr(spec, "name", None)
                or getattr(spec, "tool_name", None)
                or getattr(spec, "__name__", None)
                or fallback_name
            )
            description = str(
                getattr(spec, "description", None)
                or getattr(spec, "desc", None)
                or getattr(spec, "__doc__", None)
                or ""
            )
            parameters = (
                getattr(spec, "parameters", None)
                or getattr(spec, "params", None)
                or getattr(spec, "args_schema", None)
                or getattr(spec, "schema", None)
            )

        schema = None

        if isinstance(parameters, dict):
            schema = _jsonable(parameters)
        else:
            schema = _schema_from_param_obj(parameters)

        if not isinstance(schema, dict):
            schema = _known_tool_schema(name)

        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                },
            }
        )

    return out


def _normalize_tool_call(tc: Any) -> dict[str, Any] | None:
    if not isinstance(tc, dict):
        return None

    out: dict[str, Any] = {}

    if tc.get("id") is not None:
        out["id"] = tc.get("id")
    elif tc.get("tool_call_id") is not None:
        out["id"] = tc.get("tool_call_id")

    if tc.get("type") is not None:
        out["type"] = tc.get("type")

    fn = tc.get("function")
    if isinstance(fn, dict):
        name = fn.get("name") or tc.get("name") or "unknown"
        args = _parse_json_if_possible(fn.get("arguments", tc.get("arguments", {})))
        if not isinstance(args, dict):
            args = {"value": args}

        out["function"] = {
            "name": name,
            "arguments": args,
        }
        out["name"] = name
        out["arguments"] = args
        return out

    name = tc.get("name") or "unknown"
    args = _parse_json_if_possible(tc.get("arguments", {}))
    if not isinstance(args, dict):
        args = {"value": args}

    out["name"] = name
    out["arguments"] = args
    return out


def sanitize_messages_for_replay(messages: Any) -> list[dict[str, Any]]:
    raw = _jsonable(messages)
    if not isinstance(raw, list):
        return []

    out: list[dict[str, Any]] = []

    for msg in raw:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        m: dict[str, Any] = {"role": role}

        if msg.get("content") is not None:
            m["content"] = msg.get("content")
        else:
            m["content"] = ""

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            norm_tool_calls = []
            for tc in tool_calls:
                norm = _normalize_tool_call(tc)
                if norm is not None:
                    norm_tool_calls.append(norm)
            if norm_tool_calls:
                m["tool_calls"] = norm_tool_calls

        if role == "tool":
            if msg.get("tool_call_id") is not None:
                m["tool_call_id"] = msg.get("tool_call_id")
            if msg.get("name") is not None:
                m["name"] = msg.get("name")

        out.append(m)

    return out


def _find_fields_by_key(obj: Any, wanted_keys: set[str], path: str = "") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if str(k) in wanted_keys:
                found.append(
                    {
                        "path": p,
                        "value": _jsonable(v),
                    }
                )
            found.extend(_find_fields_by_key(v, wanted_keys, p))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_find_fields_by_key(v, wanted_keys, f"{path}[{i}]"))

    return found


def _extract_token_id_summary(obj: Any) -> dict[str, Any]:
    j = _jsonable(obj)

    fields = _find_fields_by_key(
        j,
        {
            "prompt_token_ids",
            "token_ids",
            "input_token_ids",
            "output_token_ids",
            "completion_token_ids",
            "generated_token_ids",
        },
    )

    prompt_token_ids = None
    generated_token_ids = None

    for item in fields:
        path = item["path"]
        value = item["value"]

        if path.endswith("prompt_token_ids") and isinstance(value, list):
            prompt_token_ids = value

        if path.endswith("token_ids") and "prompt_token_ids" not in path and isinstance(value, list):
            generated_token_ids = value

    return {
        "token_id_fields": fields,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "prompt_token_ids_len": len(prompt_token_ids) if isinstance(prompt_token_ids, list) else None,
        "generated_token_ids_len": len(generated_token_ids) if isinstance(generated_token_ids, list) else None,
    }


def _extract_inner_raw_token_id_summary(inner: Any) -> dict[str, Any]:
    raw = getattr(inner, "_last_raw_response_json", None)
    prompt_token_ids = getattr(inner, "_last_prompt_token_ids", None)
    generated_token_ids = getattr(inner, "_last_generated_token_ids", None)
    summary = getattr(inner, "_last_raw_response_token_id_summary", None)

    if not isinstance(summary, dict):
        summary = {}

    if not isinstance(prompt_token_ids, list) and isinstance(raw, dict):
        v = raw.get("prompt_token_ids")
        if isinstance(v, list):
            prompt_token_ids = v

    if not isinstance(generated_token_ids, list) and isinstance(raw, dict):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            v = choices[0].get("token_ids")
            if isinstance(v, list):
                generated_token_ids = v

    return {
        "source": "inner_raw_response",
        "prompt_token_ids": prompt_token_ids if isinstance(prompt_token_ids, list) else None,
        "generated_token_ids": generated_token_ids if isinstance(generated_token_ids, list) else None,
        "prompt_token_ids_len": len(prompt_token_ids) if isinstance(prompt_token_ids, list) else None,
        "generated_token_ids_len": len(generated_token_ids) if isinstance(generated_token_ids, list) else None,
        "has_inner_raw_response_json": isinstance(raw, dict),
        "inner_summary": _jsonable(summary),
    }


def _render_replay_prompt(
    *,
    model: str | None,
    chat_template_path: str | Path | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    enable_thinking: bool | str | None,
    log_input_ids: bool,
) -> dict[str, Any]:
    replay: dict[str, Any] = {
        "chat_template_path": str(chat_template_path) if chat_template_path else None,
        "rendered_with_tools": bool(tools),
    }

    if not model:
        replay["render_error"] = "model is not set"
        return replay

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        replay["render_error"] = f"{type(exc).__name__}: {exc}"
        return replay

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        if chat_template_path:
            tokenizer.chat_template = Path(chat_template_path).read_text(encoding="utf-8")

        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools

        if enable_thinking is not None:
            if isinstance(enable_thinking, str):
                if enable_thinking.lower() == "true":
                    kwargs["enable_thinking"] = True
                elif enable_thinking.lower() == "false":
                    kwargs["enable_thinking"] = False
            else:
                kwargs["enable_thinking"] = bool(enable_thinking)

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

        enc = tokenizer(
            rendered,
            add_special_tokens=False,
        )

        input_ids = enc["input_ids"]

        replay["rendered_prompt"] = rendered
        replay["input_token_count_hf"] = len(input_ids)
        replay["render_error"] = None

        if log_input_ids:
            replay["input_ids"] = input_ids

        return replay

    except Exception as exc:
        replay["render_error"] = f"{type(exc).__name__}: {exc}"
        return replay


class LoggedStirrupClient:
    def __init__(
        self,
        inner: Any,
        log_path: str | Path,
        *,
        model: str | None = None,
        chat_template_path: str | Path | None = None,
        enable_thinking: bool | str | None = None,
        request_defaults: dict[str, Any] | None = None,
        log_input_ids: bool = False,
    ):
        self.inner = inner
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.call_index = 0

        self.model = model
        self.chat_template_path = chat_template_path
        self.enable_thinking = enable_thinking
        self.request_defaults = request_defaults or {}
        self.log_input_ids = log_input_ids

    async def generate(self, messages: Any, tools: Any):
        self.call_index += 1

        raw_messages = _jsonable(messages)
        replay_messages = sanitize_messages_for_replay(messages)
        normalized_tools = normalize_logged_tools(tools)

        replay = _render_replay_prompt(
            model=self.model,
            chat_template_path=self.chat_template_path,
            messages=replay_messages,
            tools=normalized_tools,
            enable_thinking=self.enable_thinking,
            log_input_ids=self.log_input_ids,
        )

        record: dict[str, Any] = {
            "call_index": self.call_index,
            "started_at_unix": time.time(),
            "request": {
                **_jsonable(self.request_defaults),
                "messages": raw_messages,
                "messages_for_replay": replay_messages,
                "tools": normalized_tools,
                "stirrup_tools_raw": _jsonable(tools),
            },
            "replay": replay,
        }

        try:
            response = await self.inner.generate(messages, tools)

            response_json = _jsonable(response)

            response_token_id_summary = _extract_token_id_summary(response_json)
            inner_raw_token_id_summary = _extract_inner_raw_token_id_summary(self.inner)

            prompt_token_ids = inner_raw_token_id_summary.get("prompt_token_ids")
            generated_token_ids = inner_raw_token_id_summary.get("generated_token_ids")

            if not isinstance(prompt_token_ids, list):
                prompt_token_ids = response_token_id_summary.get("prompt_token_ids")

            if not isinstance(generated_token_ids, list):
                generated_token_ids = response_token_id_summary.get("generated_token_ids")

            merged_token_id_summary = {
                "response_object_summary": response_token_id_summary,
                "inner_raw_response_summary": inner_raw_token_id_summary,
                "prompt_token_ids": prompt_token_ids if isinstance(prompt_token_ids, list) else None,
                "generated_token_ids": generated_token_ids if isinstance(generated_token_ids, list) else None,
                "prompt_token_ids_len": len(prompt_token_ids) if isinstance(prompt_token_ids, list) else None,
                "generated_token_ids_len": len(generated_token_ids) if isinstance(generated_token_ids, list) else None,
            }

            record["finished_at_unix"] = time.time()
            record["response"] = response_json
            record["response_token_ids"] = merged_token_id_summary
            record["error"] = None

            record["replay"]["prompt_token_ids_from_response"] = merged_token_id_summary.get("prompt_token_ids")
            record["replay"]["generated_token_ids_from_response"] = merged_token_id_summary.get("generated_token_ids")
            record["replay"]["prompt_token_ids_from_response_len"] = merged_token_id_summary.get("prompt_token_ids_len")
            record["replay"]["generated_token_ids_from_response_len"] = merged_token_id_summary.get("generated_token_ids_len")

            return response

        except Exception as e:
            inner_raw_token_id_summary = _extract_inner_raw_token_id_summary(self.inner)

            record["finished_at_unix"] = time.time()
            record["response"] = None
            record["response_token_ids"] = {
                "inner_raw_response_summary": inner_raw_token_id_summary,
            }
            record["error"] = {
                "type": type(e).__name__,
                "message": str(e),
            }
            raise

        finally:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __getattr__(self, name: str):
        return getattr(self.inner, name)
