"""Short-term memory: the session's conversation."""

from __future__ import annotations

from typing import Any

from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.neo4j._convert import to_chat_message

# The library's get_conversation(limit=N) returns the *oldest* N messages. Read the most recent N instead,
# from the session's latest conversation (the one the library itself picks), oldest first.
_RECENT_MESSAGES = """
MATCH (c:Conversation {session_id: $session_id})
WITH c ORDER BY c.created_at DESC LIMIT 1
MATCH (c)-[:HAS_MESSAGE]->(m:Message)
WITH m ORDER BY m.timestamp DESC, m.id DESC LIMIT $limit
RETURN m.id AS id, m.role AS role, m.content AS content
ORDER BY m.timestamp ASC, m.id ASC
"""


class _Row:
    def __init__(self, row: dict[str, Any]) -> None:
        self.id, self.role, self.content = row["id"], row["role"], row["content"]


async def read_history(client: Any, session_id: str, limit: int) -> list[ChatMessage]:
    """The session's ``limit`` most recent user and assistant messages, oldest first."""
    if limit <= 0:
        return []
    rows = await client.query.cypher(_RECENT_MESSAGES, {"session_id": session_id, "limit": limit})
    history = [to_chat_message(_Row(r)) for r in rows]
    return [m for m in history if m is not None]


async def add_message(  # noqa: PLR0913
    client: Any,
    session_id: str,
    role: str,
    content: str,
    *,
    extraction_mode: str,
    mentions: list[Any] | None = None,
    user_id: str | None = None,
) -> Any:
    """Store one message. ``extraction_mode`` is the library's own: ``auto``, ``explicit`` or ``skip``."""
    kwargs: dict[str, Any] = {"extraction_mode": extraction_mode, "user_identifier": user_id}
    if extraction_mode == "explicit":
        kwargs["explicit_mentions"] = mentions or []
    return await client.short_term.add_message(session_id, role, content, **kwargs)
