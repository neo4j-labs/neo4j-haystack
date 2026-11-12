"""Agent tools for Neo4j agent memory: let the model decide when to recall and when to store memories."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from haystack import logging
from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.components.retrievers.neo4j.memory_retriever import Neo4jMemoryRetriever
from haystack_integrations.memory_stores.neo4j import _short_term
from haystack_integrations.memory_stores.neo4j.config import RecallConfig
from haystack_integrations.memory_stores.neo4j.store import Neo4jMemoryStore

logger = logging.getLogger(__name__)

_DEFAULT_INPUTS_FROM_STATE: dict[str, str] = {"session_id": "session_id", "user_id": "user_id"}

# ---------------------------------------------------------------- retrieve_memories

_RETRIEVER_DESCRIPTION = (
    "Recall memory about the user and past work. With a query, returns the user's preferences, known entities, "
    "facts and similar past tasks relevant to it. Without a query, returns this conversation's recent messages and "
    "the user's preferences."
)
_RETRIEVER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": ["string", "null"],
            "description": "What to recall. Omit or pass null for recent conversation and the user's preferences.",
        },
    },
}


class Neo4jMemoryRetrieverTool(Tool):
    """
    A tool that recalls memories from a ``Neo4jMemoryStore``, for an ``Agent`` that decides when to look.

    ``session_id`` and ``user_id`` are injected from Agent State (``inputs_from_state``), so one tool serves many
    users; the model only sees ``query``.

    Usage example:

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack_integrations.memory_stores.neo4j import Neo4jMemoryStore
    from haystack_integrations.tools.neo4j import Neo4jMemoryRetrieverTool, Neo4jMemoryWriterTool

    store = Neo4jMemoryStore()
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4-mini"),
        tools=[Neo4jMemoryRetrieverTool(memory_store=store), Neo4jMemoryWriterTool(memory_store=store)],
        state_schema={"session_id": {"type": str}, "user_id": {"type": str}},
    )
    agent.run(messages=[...], session_id="s1", user_id="pravesh")
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        memory_store: Neo4jMemoryStore,
        recall: RecallConfig | None = None,
        name: str = "retrieve_memories",
        description: str = _RETRIEVER_DESCRIPTION,
        parameters: dict[str, Any] = _RETRIEVER_PARAMETERS,
        inputs_from_state: dict[str, str] = _DEFAULT_INPUTS_FROM_STATE,
    ) -> None:
        """
        :param memory_store: The store to recall from.
        :param recall: What to read and how much. The conversation history is read only for calls without a query.
        :param name: Tool name shown to the model.
        :param description: Tool description shown to the model.
        :param parameters: JSON schema of the parameters shown to the model.
        :param inputs_from_state: Agent State keys mapped to this tool's ``session_id`` / ``user_id`` parameters.
        """
        self.memory_store = memory_store
        self.recall = recall or RecallConfig()
        # Query recall: long-term and reasoning memory; the agent already has the conversation in its messages.
        query_recall = RecallConfig(**{**self.recall.to_dict(), "read_short_term": False})
        self._retriever = Neo4jMemoryRetriever(memory_store, recall=query_recall)
        super().__init__(
            name=name,
            description=description,
            parameters=deepcopy(parameters),
            function=self.retrieve,
            inputs_from_state=dict(inputs_from_state),
        )

    def warm_up(self) -> None:
        self._retriever.warm_up()

    def retrieve(self, query: str | None = None, *, session_id: str | None = None, user_id: str | None = None) -> str:
        """
        Recall memories as text for the model.

        :param query: What to recall. ``None`` returns recent conversation and the user's preferences.
        :param session_id: Conversation session, injected from Agent State.
        :param user_id: User, injected from Agent State.
        """
        try:
            if query:
                # The read itself, not the component's run(): a failure must say "unavailable", not "nothing found".
                context = self.memory_store.run(self._retriever._read(query, session_id or ""))["context"]
                return context or "No memories found."
            return self.memory_store.run(self._overview(session_id, user_id)) or "No memories found."
        except Exception as error:
            if self.memory_store.strict:
                raise
            logger.warning("retrieve_memories failed ({error}).", error=error)
            return "Memory is unavailable right now."

    async def _overview(self, session_id: str | None, user_id: str | None) -> str:
        client = await self.memory_store._connected()
        lines: list[str] = []
        if session_id and self.recall.history_limit:
            history = await _short_term.read_history(client, session_id, self.recall.history_limit)
            if history:
                lines.append("Recent conversation:")
                lines += [f"- {m.role.value}: {m.text}" for m in history]
        if user_id and not self.memory_store.is_nams:  # NAMS stores no preferences
            preferences = await client.long_term.get_preferences_for(user_id)
            if preferences:
                lines.append("User preferences:")
                lines += [f"- [{p.category}] {p.preference}" for p in preferences[: self.recall.preference_limit]]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "memory_store": self.memory_store.to_dict(),
                "recall": self.recall.to_dict(),
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "inputs_from_state": self.inputs_from_state,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Neo4jMemoryRetrieverTool:
        inner = dict(data["data"])
        inner["memory_store"] = Neo4jMemoryStore.from_dict(inner["memory_store"])
        inner["recall"] = RecallConfig.from_dict(inner["recall"])
        return cls(**inner)


# ---------------------------------------------------------------- store_memory

_WRITER_DESCRIPTION = (
    "Store durable information about the user for future conversations: a preference (how they like things), a "
    "fact (subject, predicate, object), or an entity (a person, organization, place, event or object). Do not "
    "store things that only matter for the current conversation."
)
_WRITER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memory_type": {"type": "string", "enum": ["preference", "fact", "entity"], "description": "What to store."},
        "category": {"type": "string", "description": "preference: a short topic, e.g. 'region' or 'style'."},
        "preference": {"type": "string", "description": "preference: the preference, e.g. 'Focus on Germany'."},
        "subject": {"type": "string", "description": "fact: who or what the fact is about."},
        "predicate": {"type": "string", "description": "fact: the relation, e.g. 'works_at'."},
        "object": {"type": "string", "description": "fact: the value, e.g. 'Northwind Capital'."},
        "name": {"type": "string", "description": "entity: its name."},
        "entity_type": {
            "type": "string",
            "enum": ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "OBJECT"],
            "description": "entity: its type.",
        },
        "description": {"type": "string", "description": "entity: optional short description."},
    },
    "required": ["memory_type"],
}
_REQUIRED_FIELDS = {
    "preference": ("category", "preference"),
    "fact": ("subject", "predicate", "object"),
    "entity": ("name", "entity_type"),
}


class Neo4jMemoryWriterTool(Tool):
    """
    A tool that stores a preference, fact or entity in a ``Neo4jMemoryStore``, when the model decides to.

    ``user_id`` is injected from Agent State. Missing fields and memory types the backend cannot store (NAMS stores
    no preferences or facts) are reported back to the model as text instead of failing the run.
    """

    def __init__(
        self,
        *,
        memory_store: Neo4jMemoryStore,
        name: str = "store_memory",
        description: str = _WRITER_DESCRIPTION,
        parameters: dict[str, Any] = _WRITER_PARAMETERS,
        inputs_from_state: dict[str, str] = _DEFAULT_INPUTS_FROM_STATE,
    ) -> None:
        """
        :param memory_store: The store to write to.
        :param name: Tool name shown to the model.
        :param description: Tool description shown to the model.
        :param parameters: JSON schema of the parameters shown to the model.
        :param inputs_from_state: Agent State keys mapped to this tool's ``session_id`` / ``user_id`` parameters.
        """
        self.memory_store = memory_store
        super().__init__(
            name=name,
            description=description,
            parameters=deepcopy(parameters),
            function=self.store,
            inputs_from_state=dict(inputs_from_state),
        )

    def warm_up(self) -> None:
        try:
            self.memory_store.warm_up()
        except Exception as error:
            if self.memory_store.strict:
                raise
            logger.warning("Neo4j memory is unavailable ({error}); continuing without it.", error=error)

    def store(  # noqa: PLR0913, PLR0917 - the model's JSON fields
        self,
        memory_type: str,
        category: str | None = None,
        preference: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        name: str | None = None,
        entity_type: str | None = None,
        description: str | None = None,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Store one memory and tell the model what happened."""
        values = {"category": category, "preference": preference, "subject": subject, "predicate": predicate,
                  "object": object, "name": name, "entity_type": entity_type}  # fmt: skip
        if memory_type not in _REQUIRED_FIELDS:
            return f"Not stored: memory_type must be one of {sorted(_REQUIRED_FIELDS)}."
        missing = [field for field in _REQUIRED_FIELDS[memory_type] if not values[field]]
        if missing:
            return f"Not stored: a {memory_type} needs {', '.join(missing)}."
        try:
            if memory_type == "preference":
                saved = self.memory_store.add_preference(category, preference, user_id=user_id)
                return f"Stored preference [{category}] {preference}." if saved else _not_on_nams("preferences")
            if memory_type == "fact":
                saved = self.memory_store.add_fact(subject, predicate, object)
                return f"Stored fact: {subject} {predicate} {object}." if saved else _not_on_nams("facts")
            entity = self.memory_store.add_entity(name, entity_type, description=description)
            merged = f" (matched existing entity {entity['matched']})" if entity.get("resolution") == "merged" else ""
            return f"Stored entity {entity['name']} ({entity['type']}){merged}."
        except Exception as error:
            if self.memory_store.strict:
                raise
            logger.warning("store_memory failed ({error}).", error=error)
            return "Not stored: memory is unavailable right now."

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "memory_store": self.memory_store.to_dict(),
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "inputs_from_state": self.inputs_from_state,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Neo4jMemoryWriterTool:
        inner = dict(data["data"])
        inner["memory_store"] = Neo4jMemoryStore.from_dict(inner["memory_store"])
        return cls(**inner)


def _not_on_nams(what: str) -> str:
    return f"Not stored: this memory backend (NAMS) does not store {what}."
