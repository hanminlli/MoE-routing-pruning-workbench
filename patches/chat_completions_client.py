"""OpenAI SDK-based LLM client for chat completions.

Patched for the Qwen/vLLM GDPval routing experiment:
- Preserves the raw OpenAI/vLLM response on the client object.
- Preserves vLLM token-id fields when returned:
  - prompt_token_ids
  - choices[0].token_ids
- Keeps exact token accounting possible.
- Handles finish_reason="length" without deterministic retry loops.

Important length-handling behavior:
- If vLLM returns finish_reason="length" but also returns parsed tool calls,
  return those tool calls to Stirrup.
- If vLLM returns finish_reason="length" with no parsed tool calls, return a
  short synthetic assistant message instead of raising ContextOverflowError.
  This prevents Stirrup from retrying from the same state forever at temperature=0.
"""

import logging
import os
from time import perf_counter
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from stirrup.clients.utils import to_openai_messages, to_openai_tools
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    LLMClient,
    Reasoning,
    TokenUsage,
    Tool,
    ToolCall,
)

__all__ = ["ChatCompletionsClient"]

LOGGER = logging.getLogger(__name__)


class ChatCompletionsClient(LLMClient):
    def __init__(
        self,
        model: str,
        max_tokens: int = 64_000,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._kwargs = kwargs or {}

        self._last_raw_response = None
        self._last_raw_response_json: dict[str, Any] | None = None
        self._last_prompt_token_ids: list[int] | None = None
        self._last_generated_token_ids: list[int] | None = None
        self._last_raw_response_token_id_summary: dict[str, Any] | None = None

        resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def model_slug(self) -> str:
        return self._model

    def _reset_last_raw_response_fields(self) -> None:
        self._last_raw_response = None
        self._last_raw_response_json = None
        self._last_prompt_token_ids = None
        self._last_generated_token_ids = None
        self._last_raw_response_token_id_summary = None

    def _store_raw_response_fields(self, response: Any) -> None:
        """
        Store raw response fields before converting to Stirrup's AssistantMessage.

        vLLM returns token IDs when the request includes:
            extra_body={"return_token_ids": True}

        Observed vLLM response shape:
            response.prompt_token_ids
            response.choices[0].token_ids
        """
        self._last_raw_response = response

        raw: dict[str, Any] | None = None

        if hasattr(response, "model_dump"):
            try:
                raw = response.model_dump(mode="json")
            except TypeError:
                try:
                    raw = response.model_dump()
                except Exception:
                    raw = None
            except Exception:
                raw = None

        if raw is None and hasattr(response, "dict"):
            try:
                raw = response.dict()
            except Exception:
                raw = None

        if raw is None:
            raw = {}

        self._last_raw_response_json = raw

        prompt_token_ids = raw.get("prompt_token_ids")
        generated_token_ids = None

        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                generated_token_ids = first_choice.get("token_ids")

        if not isinstance(prompt_token_ids, list):
            prompt_token_ids = None

        if not isinstance(generated_token_ids, list):
            generated_token_ids = None

        self._last_prompt_token_ids = prompt_token_ids
        self._last_generated_token_ids = generated_token_ids

        self._last_raw_response_token_id_summary = {
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": generated_token_ids,
            "prompt_token_ids_len": len(prompt_token_ids) if isinstance(prompt_token_ids, list) else None,
            "generated_token_ids_len": len(generated_token_ids) if isinstance(generated_token_ids, list) else None,
            "has_raw_response_json": bool(raw),
        }

    @retry(
        retry=retry_if_exception_type(
            (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                InternalServerError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(
        self,
        messages: list[ChatMessage],
        tools: dict[str, Tool],
    ) -> AssistantMessage:
        self._reset_last_raw_response_fields()

        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": to_openai_messages(messages),
            "max_completion_tokens": self._max_tokens,
            **self._kwargs,
        }

        if tools:
            request_kwargs["tools"] = to_openai_tools(tools)
            request_kwargs["tool_choice"] = "auto"

        if self._reasoning_effort:
            request_kwargs["reasoning_effort"] = self._reasoning_effort

        request_start_time = perf_counter()
        response = await self._client.chat.completions.create(**request_kwargs)
        request_end_time = perf_counter()

        self._store_raw_response_fields(response)

        choice = response.choices[0]
        msg = choice.message

        reasoning: Reasoning | None = None

        reasoning_content = None
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            reasoning_content = msg.reasoning_content
        elif hasattr(msg, "reasoning") and msg.reasoning:
            reasoning_content = msg.reasoning

        if reasoning_content:
            reasoning = Reasoning(content=reasoning_content)

        tool_calls = [
            ToolCall(
                tool_call_id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments or "",
            )
            for tc in (msg.tool_calls or [])
        ]

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        reasoning_tokens = 0
        if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0

        answer_tokens = output_tokens - reasoning_tokens

        content = msg.content or ""

        if choice.finish_reason in ("max_tokens", "length"):
            if tool_calls:
                LOGGER.warning(
                    "Model %s returned finish_reason=%s but also returned %d parsed tool call(s); "
                    "returning tool call(s) instead of raising ContextOverflowError.",
                    self.model_slug,
                    choice.finish_reason,
                    len(tool_calls),
                )
            else:
                LOGGER.warning(
                    "Model %s returned finish_reason=%s with no parsed tool calls; "
                    "returning synthetic truncation feedback instead of raising ContextOverflowError.",
                    self.model_slug,
                    choice.finish_reason,
                )

                content = (
                    "MODEL_OUTPUT_TRUNCATED_BY_MAX_TOKENS: The previous model response reached "
                    "the maximum completion-token limit before producing a usable tool call. "
                    "Do not repeat the same search or the same command. If recent web/API/stock-media "
                    "searches failed, returned irrelevant results, or required unavailable credentials, "
                    "stop searching that source. Proceed with a simpler approach: create the requested "
                    "deliverable using local files, generated visuals, simple placeholders, or available "
                    "standard libraries. Then verify the deliverable exists and call finish."
                )

        # Preserve the old hard failure only for a truly unusable max-token response
        # when there are no tools available at all. In normal GDPval runs, tools exist,
        # so the synthetic feedback above lets the agent move forward instead of
        # retrying from the exact same state.
        if choice.finish_reason in ("max_tokens", "length") and not tools and not tool_calls:
            raise ContextOverflowError(
                f"Maximal context window tokens reached for model {self.model_slug}, "
                f"resulting in finish reason: {choice.finish_reason}. "
                "The response did not contain a parsed tool call and no tools are available."
            )

        return AssistantMessage(
            reasoning=reasoning,
            content=content,
            tool_calls=tool_calls,
            token_usage=TokenUsage(
                input=input_tokens,
                answer=answer_tokens,
                reasoning=reasoning_tokens,
            ),
            request_start_time=request_start_time,
            request_end_time=request_end_time,
        )
