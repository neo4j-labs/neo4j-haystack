"""Neo4jMemoryWriter: writes a turn to short-term, long-term and reasoning memory after the LLM."""

from __future__ import annotations

from typing import Any, Literal, get_args

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.neo4j import _long_term, _reasoning
from haystack_integrations.memory_stores.neo4j._convert import split_turn
from haystack_integrations.memory_stores.neo4j.config import ExtractionPolicy
from haystack_integrations.memory_stores.neo4j.store import Neo4jMemoryStore

logger = logging.getLogger(__name__)

RecordReasoning = Literal["always", "with_tools", "never"]


@component
class Neo4jMemoryWriter:
    """
    Writes one conversation turn to all three memories.

    - Short-term: the user query and the answer, as messages in the session.
    - Long-term: entities and relations under the ``ExtractionPolicy``, plus ``preferences``, ``facts`` and
      ``mentions`` passed in.
    - Reasoning: a trace for the turn (task, tool-call steps, outcome, success).

    Usage example:

    ```python
    from haystack_integrations.components.writers.neo4j import Neo4jMemoryWriter
    from haystack_integrations.memory_stores.neo4j import ExtractionPolicy, Neo4jMemoryStore

    writer = Neo4jMemoryWriter(Neo4jMemoryStore(), extraction_policy=ExtractionPolicy(user="review"))
    ```
    """

    def __init__(
        self,
        store: Neo4jMemoryStore,
        *,
        extraction_policy: ExtractionPolicy | None = None,
        write_short_term: bool = True,
        record_reasoning: RecordReasoning = "always",
    ) -> None:
        """
        :param store: The memory store. Share it with the ``Neo4jMemoryRetriever``.
        :param extraction_policy: Who decides what becomes an entity. See ``ExtractionPolicy``.
        :param write_short_term: Store the query and answer as session messages (and extract from them).
        :param record_reasoning: ``"always"`` records a trace per turn, ``"with_tools"`` only for turns with tool
            calls, ``"never"`` disables reasoning memory.
        """
        if not isinstance(store, Neo4jMemoryStore):
            msg = f"store must be a Neo4jMemoryStore, got {type(store).__name__}."
            raise ValueError(msg)
        if record_reasoning not in get_args(RecordReasoning):
            msg = f"record_reasoning must be one of {list(get_args(RecordReasoning))}, got {record_reasoning!r}."
            raise ValueError(msg)
        self._store = store
        self.extraction_policy = extraction_policy or ExtractionPolicy()
        self.write_short_term = write_short_term
        self.record_reasoning = record_reasoning

    @property
    def store(self) -> Neo4jMemoryStore:
        return self._store

    def warm_up(self) -> None:
        """Connect the store. With ``strict=False``, a failure is logged and retried on the next run."""
        try:
            self._store.warm_up()
        except Exception as error:
            if self._store.strict:
                raise
            logger.warning("Neo4j memory is unavailable ({error}); continuing without it.", error=error)

    # ------------------------------------------------------------------ run

    @component.output_types(messages_written=int, trace_id=str | None, entities=list[dict])
    def run(  # noqa: PLR0913, PLR0917 - inputs are the component's sockets
        self,
        query: str,
        session_id: str,
        replies: list[ChatMessage] | None = None,
        messages: list[ChatMessage] | None = None,
        user_id: str | None = None,
        preferences: list[dict[str, Any]] | None = None,
        facts: list[dict[str, Any]] | None = None,
        mentions: list[dict[str, Any]] | None = None,
        success: bool | None = None,
    ) -> dict[str, Any]:
        """
        Write the turn to memory.

        :param query: The user's message for this turn.
        :param session_id: The conversation session.
        :param replies: A chat generator's replies.
        :param messages: A full transcript (e.g. an Agent's ``messages``), with tool calls and results. Only the
            messages after the last user message are written, so injected history is not saved twice.
        :param user_id: Optional user, for per-user scoping.
        :param preferences: Preferences to store: ``{"category", "preference", "context"?}``.
        :param facts: Facts to store: ``{"subject", "predicate", "object"}``.
        :param mentions: Entities to link for roles whose policy mode is ``"explicit"``: ``{"name", "type"}``,
            or ``{"id"}`` for an existing entity.
        :param success: Whether the turn succeeded, for the reasoning trace. If ``None``, it's ``True`` when
            there is an answer and no tool reported an error. Pass a real signal in production.
        :returns: A dictionary with ``messages_written``, ``trace_id`` and ``entities`` (linked entities, each
            with ``name``, ``type``, ``source`` and ``role``).
        """
        self._validate(preferences, facts, mentions)
        progress = self._progress()
        try:
            return self._store.run(
                self._write(
                    query, session_id, replies, messages, user_id, preferences, facts, mentions, success, progress
                )
            )
        except Exception as error:
            return self._fallback(error, progress)

    @component.output_types(messages_written=int, trace_id=str | None, entities=list[dict])
    async def run_async(  # noqa: PLR0913, PLR0917 - inputs are the component's sockets
        self,
        query: str,
        session_id: str,
        replies: list[ChatMessage] | None = None,
        messages: list[ChatMessage] | None = None,
        user_id: str | None = None,
        preferences: list[dict[str, Any]] | None = None,
        facts: list[dict[str, Any]] | None = None,
        mentions: list[dict[str, Any]] | None = None,
        success: bool | None = None,
    ) -> dict[str, Any]:
        """Async version of ``run``."""
        self._validate(preferences, facts, mentions)
        progress = self._progress()
        try:
            return await self._store.arun(
                self._write(
                    query, session_id, replies, messages, user_id, preferences, facts, mentions, success, progress
                )
            )
        except Exception as error:
            return self._fallback(error, progress)

    @staticmethod
    def _validate(preferences: Any, facts: Any, mentions: Any) -> None:
        # Invalid items are a caller bug: always raise, even when strict=False.
        _long_term.validate_preferences(preferences)
        _long_term.validate_facts(facts)
        _long_term.validate_mentions(mentions)

    @staticmethod
    def _progress() -> dict[str, Any]:
        return {"messages_written": 0, "trace_id": None, "entities": []}

    def _fallback(self, error: Exception, progress: dict[str, Any]) -> dict[str, Any]:
        if self._store.strict:
            raise error
        logger.warning(
            "Writing to Neo4j memory failed ({error}); {written} messages were written before the failure.",
            error=error,
            written=progress["messages_written"],
        )
        return progress

    async def _write(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        session_id: str,
        replies: list[ChatMessage] | None,
        messages: list[ChatMessage] | None,
        user_id: str | None,
        preferences: list[dict[str, Any]] | None,
        facts: list[dict[str, Any]] | None,
        mentions: list[dict[str, Any]] | None,
        success: bool | None,
        progress: dict[str, Any],
    ) -> dict[str, Any]:
        client = await self._store._connected()
        policy = self.extraction_policy
        strict = self._store.strict

        # 1. The turn: messages after the query, plus the generator's replies.
        turn, answer = split_turn(messages, replies)

        # 2. Short-term messages, with long-term extraction under the policy.
        if self.write_short_term:
            for role, text, mode in (
                ("user", query, policy.user),
                ("assistant", answer.text if answer else None, policy.assistant),
            ):
                if not text:
                    continue
                _, linked = await _long_term.write_with_policy(
                    client,
                    session_id=session_id,
                    role=role,
                    text=text,
                    mode=mode,
                    policy=policy,
                    mentions=mentions,
                    turn=turn,
                    user_id=user_id,
                    strict=strict,
                )
                progress["messages_written"] += 1
                progress["entities"].extend(linked)

        # 3. Explicit long-term writes.
        if preferences:
            await _long_term.add_preferences(client, preferences, user_id)
        if facts:
            await _long_term.add_facts(client, facts)

        # 4. Reasoning trace, written in one go after the turn.
        tools_used = _reasoning.has_tool_calls(turn)
        if self.record_reasoning == "always" or (self.record_reasoning == "with_tools" and tools_used):
            progress["trace_id"] = await _reasoning.record_trace(
                client,
                session_id=session_id,
                task=query,
                turn=turn,
                answer=answer,
                success=success if success is not None else _reasoning.default_success(turn, answer),
                user_id=user_id,
            )

        logger.debug(
            "Wrote {written} messages, {entities} entities, trace {trace_id}.",
            written=progress["messages_written"],
            entities=len(progress["entities"]),
            trace_id=progress["trace_id"],
        )
        return progress

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            store=self._store.to_dict(),
            extraction_policy=self.extraction_policy.to_dict(),
            write_short_term=self.write_short_term,
            record_reasoning=self.record_reasoning,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Neo4jMemoryWriter:
        params = data.get("init_parameters", {})
        params["store"] = Neo4jMemoryStore.from_dict(params["store"])
        if params.get("extraction_policy") is not None:
            params["extraction_policy"] = ExtractionPolicy.from_dict(params["extraction_policy"])
        return default_from_dict(cls, data)
