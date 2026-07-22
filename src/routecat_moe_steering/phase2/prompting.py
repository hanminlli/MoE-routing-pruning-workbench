from __future__ import annotations

from typing import Any


def render_prompt(
    tokenizer: Any,
    text: str,
    *,
    use_chat_template: bool = True,
    system_prompt: str | None = None,
) -> str:
    """Render one instruction with the checkpoint tokenizer's chat policy.

    Raw text remains available for datasets that already contain a complete serialized
    conversation. The default is checkpoint-local chat rendering so full and pruned
    checkpoints receive the same conversational structure.
    """
    if not use_chat_template:
        return text
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("tokenizer does not expose apply_chat_template")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template did not return text")
    return rendered


def render_prompts(
    tokenizer: Any,
    texts: list[str],
    *,
    use_chat_template: bool = True,
    system_prompt: str | None = None,
) -> list[str]:
    return [
        render_prompt(
            tokenizer,
            text,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )
        for text in texts
    ]
