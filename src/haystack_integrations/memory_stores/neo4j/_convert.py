"""Conversions between Haystack ChatMessages and neo4j-agent-memory models."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from haystack.dataclasses import ChatMessage, ChatRole, ToolCallResult

MEMORY_META_KEY = "neo4j_memory"


def role_value(role: Any) -> str:
    """``MessageRole.USER`` / ``"user"`` -> ``"user"``."""
    return str(getattr(role, "value", role)).lower()


def to_chat_message(message: Any) -> ChatMessage | None:
    """Convert a stored memory message to a ChatMessage. System and tool messages are not replayed."""
    role = role_value(message.role)
    meta = {MEMORY_META_KEY: "history", "message_id": str(message.id)}
    if role == "user":
        return ChatMessage.from_user(message.content, meta=meta)
    if role == "assistant":
        return ChatMessage.from_assistant(message.content, meta=meta)
    return None


def result_text(result: ToolCallResult) -> str:
    """The text of a tool result, which may be a string or a list of content parts."""
    if isinstance(result.result, str):
        return result.result
    return "\n".join(getattr(part, "text", "") for part in result.result if getattr(part, "text", None))


def one_line(text: str | None, max_chars: int = 150) -> str:
    """Collapse whitespace and cut to ``max_chars``."""
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= max_chars else collapsed[: max_chars - 1].rstrip() + "…"


def _collect_named_objects(value: Any, found: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        name, etype = value.get("name"), value.get("type")
        if isinstance(name, str) and name.strip() and isinstance(etype, str) and etype.strip():
            found.append({"name": name.strip(), "type": etype.strip().upper()})
        for child in value.values():
            _collect_named_objects(child, found)
    elif isinstance(value, list):
        for child in value:
            _collect_named_objects(child, found)


def entities_from_tool_results(messages: Iterable[ChatMessage]) -> list[dict[str, str]]:
    """
    Collect ``{"name", "type"}`` objects from JSON tool results, at any nesting depth.

    Non-JSON results are ignored. Duplicates (same name and type, case-insensitive) are dropped.
    """
    found: list[dict[str, str]] = []
    for message in messages:
        for result in message.tool_call_results:
            if result.error:
                continue
            try:
                payload = json.loads(result_text(result))
            except (TypeError, ValueError):
                continue
            _collect_named_objects(payload, found)

    unique: dict[tuple[str, str], dict[str, str]] = {}
    for entity in found:
        unique.setdefault((entity["name"].lower(), entity["type"]), entity)
    return list(unique.values())


def split_turn(
    messages: list[ChatMessage] | None, replies: list[ChatMessage] | None
) -> tuple[list[ChatMessage], ChatMessage | None]:
    """
    Return the messages of this turn and its final answer.

    When ``messages`` is a full transcript (e.g. an Agent's output), only what follows the last user message
    belongs to this turn; everything before it is the system prompt, recalled memory and history that the
    memory retriever injected.
    """
    kept: list[ChatMessage] = []
    if messages:
        last_user = max((i for i, m in enumerate(messages) if m.is_from(ChatRole.USER)), default=-1)
        kept = list(messages[last_user + 1 :])
    turn = kept + list(replies or [])
    answer = next((m for m in reversed(turn) if m.is_from(ChatRole.ASSISTANT) and m.text), None)
    return turn, answer


def to_json_safe(value: Any) -> Any:
    """Make memory model values (UUIDs, datetimes, enums) JSON-serializable."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):  # enums
        return to_json_safe(value.value)
    return str(value)
