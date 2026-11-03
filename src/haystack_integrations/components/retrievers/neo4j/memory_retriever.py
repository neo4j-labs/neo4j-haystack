"""Neo4jMemoryRetriever: reads short-term, long-term and reasoning memory before the LLM."""

from __future__ import annotations

import asyncio
from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage
from jinja2.sandbox import SandboxedEnvironment

from haystack_integrations.memory_stores.neo4j import _long_term, _reasoning, _short_term
from haystack_integrations.memory_stores.neo4j._convert import MEMORY_META_KEY, to_json_safe
from haystack_integrations.memory_stores.neo4j.config import RecallConfig
from haystack_integrations.memory_stores.neo4j.store import Neo4jMemoryStore

logger = logging.getLogger(__name__)


@component
class Neo4jMemoryRetriever:
    """
    Reads all three memories for a query and returns prompt-ready messages.

    - Short-term: the session's recent messages.
    - Long-term: preferences, entities and facts relevant to the query.
    - Reasoning: similar past tasks that succeeded, with the tools they used.

    Connect ``messages`` straight to a chat generator, or use ``context`` in your own prompt.

    Usage example:

    ```python
    from haystack_integrations.components.retrievers.neo4j import Neo4jMemoryRetriever
    from haystack_integrations.memory_stores.neo4j import Neo4jMemoryStore

    retriever = Neo4jMemoryRetriever(Neo4jMemoryStore(), system_prompt="You are a research assistant.")
    result = retriever.run(query="What do I prefer?", session_id="s1")
    ```
    """

    def __init__(
        self,
        store: Neo4jMemoryStore,
        *,
        recall: RecallConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """
        :param store: The memory store. Share it with the ``Neo4jMemoryWriter``.
        :param recall: What to read and how much. See ``RecallConfig``.
        :param system_prompt: Optional system message placed first in ``messages``.
        """
        if not isinstance(store, Neo4jMemoryStore):
            msg = f"store must be a Neo4jMemoryStore, got {type(store).__name__}."
            raise ValueError(msg)
        self._store = store
        self.recall = recall or RecallConfig()
        self.system_prompt = system_prompt
        self._template = SandboxedEnvironment().from_string(self.recall.template)

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

    @component.output_types(messages=list[ChatMessage], history=list[ChatMessage], context=str, memories=dict)
    def run(self, query: str, session_id: str) -> dict[str, Any]:
        """
        Read memory for ``query``.

        :param query: The user's message for this turn.
        :param session_id: The conversation session.
        :returns: A dictionary with:
            - ``messages``: system prompt, recalled memory (if any), history, then the query as a user message.
            - ``history``: short-term messages only.
            - ``context``: rendered long-term and reasoning recall; empty if nothing was found.
            - ``memories``: ``{"preferences", "entities", "facts", "traces"}`` with ids and similarity scores.
        """
        try:
            return self._store.run(self._read(query, session_id))
        except Exception as error:
            return self._fallback(query, error)

    @component.output_types(messages=list[ChatMessage], history=list[ChatMessage], context=str, memories=dict)
    async def run_async(self, query: str, session_id: str) -> dict[str, Any]:
        """Async version of ``run``."""
        try:
            return await self._store.arun(self._read(query, session_id))
        except Exception as error:
            return self._fallback(query, error)

    def _fallback(self, query: str, error: Exception) -> dict[str, Any]:
        if self._store.strict:
            raise error
        logger.warning("Reading Neo4j memory failed ({error}); continuing without memory.", error=error)
        empty = {"preferences": [], "entities": [], "facts": [], "traces": []}
        return self._result(query, history=[], memories=empty)

    async def _read(self, query: str, session_id: str) -> dict[str, Any]:
        client = await self._store._connected()
        cfg = self.recall

        async def nothing() -> Any:
            return None

        history, long_term, traces = await asyncio.gather(
            _short_term.read_history(client, session_id, cfg.history_limit) if cfg.read_short_term else nothing(),
            _long_term.recall(client, query, cfg) if cfg.read_long_term else nothing(),
            _reasoning.recall_traces(client, query, cfg) if cfg.read_reasoning else nothing(),
        )
        long_term = long_term or {"preferences": [], "entities": [], "facts": []}
        memories = {**long_term, "traces": traces or []}
        logger.debug(
            "Recalled {history} history messages, {prefs} preferences, {entities} entities, {facts} facts, "
            "{traces} traces.",
            history=len(history or []),
            prefs=len(memories["preferences"]),
            entities=len(memories["entities"]),
            facts=len(memories["facts"]),
            traces=len(memories["traces"]),
        )
        return self._result(query, history=history or [], memories=memories)

    def _result(self, query: str, *, history: list[ChatMessage], memories: dict[str, list[Any]]) -> dict[str, Any]:
        recalled = any(memories[k] for k in ("preferences", "entities", "facts", "traces"))
        context = self._template.render(**memories).strip() if recalled else ""

        messages: list[ChatMessage] = []
        if self.system_prompt:
            messages.append(ChatMessage.from_system(self.system_prompt))
        if context:
            messages.append(ChatMessage.from_system(context, meta={MEMORY_META_KEY: "recall"}))
        messages.extend(history)
        messages.append(ChatMessage.from_user(query))
        return {"messages": messages, "history": history, "context": context, "memories": to_json_safe(memories)}

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self, store=self._store.to_dict(), recall=self.recall.to_dict(), system_prompt=self.system_prompt
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Neo4jMemoryRetriever:
        params = data.get("init_parameters", {})
        params["store"] = Neo4jMemoryStore.from_dict(params["store"])
        if params.get("recall") is not None:
            params["recall"] = RecallConfig.from_dict(params["recall"])
        return default_from_dict(cls, data)
