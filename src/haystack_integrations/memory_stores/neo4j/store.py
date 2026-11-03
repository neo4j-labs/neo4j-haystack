"""Neo4jMemoryStore: connection, configuration and serialization shared by the memory components."""

from __future__ import annotations

import asyncio
import copy
import json
import logging as std_logging
import os
import uuid
import weakref
from collections.abc import Coroutine
from typing import Any, Literal, TypeVar, get_args

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.memory_stores.neo4j import _long_term, _reasoning
from haystack_integrations.memory_stores.neo4j._convert import split_turn
from haystack_integrations.memory_stores.neo4j._embedding import _SharedEmbedder
from haystack_integrations.memory_stores.neo4j._loop import _LoopRunner
from haystack_integrations.memory_stores.neo4j.config import ExtractionPolicy

_INSTALL_HINT = "Run 'pip install \"haystack-neo4j[memory]\"' to use the Neo4j agent memory components"

with LazyImport(message=_INSTALL_HINT) as memory_import:
    from neo4j_agent_memory import (
        ExtractionConfig,
        MemoryClient,
        MemoryConfig,
        MemorySettings,
        NamsConfig,
        Neo4jConfig,
        ResolutionConfig,
    )
    from neo4j_agent_memory.embeddings.base import adapt_to_legacy_embedder
    from pydantic import SecretStr

logger = logging.getLogger(__name__)
T = TypeVar("T")

Backend = Literal["bolt", "nams"]
# NamsConfig fields set through dedicated parameters; secrets must never go in the plain `nams` dict.
_NAMS_RESERVED = {"api_key", "workspace_id", "endpoint", "headers"}

# Pipeline YAML serializes the store once per component. Loading returns the live store for an identical
# config, so a loaded pipeline shares one connection. Keys hold env-var names only, never secret values.
_REGISTRY: weakref.WeakValueDictionary[str, Neo4jMemoryStore] = weakref.WeakValueDictionary()


class Neo4jMemoryStore:
    """
    Neo4j-backed short-term, long-term and reasoning memory, built on ``neo4j-agent-memory``.

    Holds the connection and configuration. ``Neo4jMemoryRetriever`` and ``Neo4jMemoryWriter`` both take a
    store, so they share one connection. No network call is made until ``warm_up()`` or first use.

    Two backends, as in ``neo4j-agent-memory``:

    - ``backend="bolt"`` (default): the library connects to your Neo4j database directly. All memory types.
    - ``backend="nams"``: the library calls the hosted Neo4j Agent Memory Service (NAMS), which stores memory
      in your workspace and extracts entities server-side. NAMS assigns session (conversation) ids: get one
      with ``create_session()``. NAMS does not support preferences, facts, relations or similar-task recall.

    Usage example:

    ```python
    from haystack_integrations.memory_stores.neo4j import Neo4jMemoryStore

    store = Neo4jMemoryStore()  # bolt: reads NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    nams_store = Neo4jMemoryStore(backend="nams")  # reads NAMS_API_KEY, NAMS_WORKSPACE_ID
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        backend: Backend = "bolt",
        uri: Secret = Secret.from_env_var("NEO4J_URI"),  # noqa: B008
        username: Secret = Secret.from_env_var("NEO4J_USERNAME", strict=False),  # noqa: B008
        password: Secret = Secret.from_env_var("NEO4J_PASSWORD"),  # noqa: B008
        database: str | None = None,
        llm_model: str = "openai/gpt-5.4-mini",
        embedding_model: str = "openai/text-embedding-3-small",
        extraction: dict[str, Any] | None = None,
        resolution: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        nams_api_key: Secret = Secret.from_env_var("NAMS_API_KEY", strict=False),  # noqa: B008
        nams_workspace_id: str | None = None,
        nams_endpoint: str | None = None,
        nams: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> None:
        """
        Create the store. Settings are validated here; nothing connects until first use.

        :param backend: ``"bolt"`` (direct Neo4j connection) or ``"nams"`` (hosted Neo4j Agent Memory Service).
        :param uri: Neo4j URI as a ``Secret`` (e.g. ``bolt://localhost:7687``).
        :param username: Neo4j username as a ``Secret``. Defaults to ``neo4j`` when the variable is unset.
        :param password: Neo4j password as a ``Secret``.
        :param database: Neo4j database name. If ``None`` (default), the ``NEO4J_DATABASE`` environment variable is
            used, else ``"neo4j"``. Hosted and shared Neo4j accounts often may only use their own database, so set
            this (or ``NEO4J_DATABASE``) to that database's name.
        :param llm_model: Provider string for extraction, e.g. ``"openai/gpt-5.4-mini"``.
        :param embedding_model: Provider string for embeddings, e.g. ``"openai/text-embedding-3-small"``.
        :param extraction: ``neo4j-agent-memory`` ``ExtractionConfig`` fields. Defaults to
            ``{"extractor_type": "llm"}``, which needs no model download. Other extractor types: ``"pipeline"``
            (spaCy, GLiNER, then LLM), ``"gliner"``, ``"spacy"``, ``"none"``.
        :param resolution: ``ResolutionConfig`` fields (entity de-duplication).
        :param memory: ``MemoryConfig`` fields, e.g. ``{"multi_tenant": True}``.
        :param nams_api_key: NAMS: API key as a ``Secret``. Defaults to the ``NAMS_API_KEY`` environment variable.
        :param nams_workspace_id: NAMS: workspace id. If ``None``, the ``NAMS_WORKSPACE_ID`` environment variable.
        :param nams_endpoint: NAMS: API endpoint. If ``None``, the library default (the hosted service).
        :param nams: NAMS: other ``NamsConfig`` fields, e.g. ``{"timeout": 60, "validate_on_connect": False}``.
            Not for credentials: ``api_key``, ``workspace_id``, ``endpoint`` and ``headers`` are rejected here.
        :param strict: If ``False`` (default), memory failures are logged and pipelines keep running.
            If ``True``, they raise.
        """
        memory_import.check()
        if backend not in get_args(Backend):
            msg = f"backend must be one of {list(get_args(Backend))}, got {backend!r}."
            raise ValueError(msg)
        reserved = _NAMS_RESERVED & set(nams or {})
        if reserved:
            msg = (
                f"Set {sorted(reserved)} through nams_api_key / nams_workspace_id / nams_endpoint, not the `nams` "
                "dict (it is saved to YAML in plain text)."
            )
            raise ValueError(msg)
        self.backend = backend
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.extraction = extraction
        self.resolution = resolution
        self.memory = memory
        self.nams_api_key = nams_api_key
        self.nams_workspace_id = nams_workspace_id
        self.nams_endpoint = nams_endpoint
        self.nams = nams
        self.strict = strict

        # Validate the library configs now, so a typo fails at construction and not mid-pipeline.
        ExtractionConfig(**self._extraction_fields())
        ResolutionConfig(**(resolution or {}))
        MemoryConfig(**(memory or {}))
        NamsConfig(**(nams or {}))

        self._runner = _LoopRunner()
        self._client: Any | None = None
        self._connect_lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------ settings and connection

    def _extraction_fields(self) -> dict[str, Any]:
        return dict(self.extraction) if self.extraction is not None else {"extractor_type": "llm"}

    @property
    def is_nams(self) -> bool:
        return self.backend == "nams"

    def _settings(self) -> MemorySettings:
        """Build settings explicitly, so behaviour never depends on NAM_* variables or a local .env file."""
        if self.is_nams:
            api_key = self.nams_api_key.resolve_value()
            if not api_key:
                msg = "backend='nams' needs an API key: set NAMS_API_KEY or pass nams_api_key=Secret.from_env_var(...)."
                raise ValueError(msg)
            fields: dict[str, Any] = {
                **(self.nams or {}),
                "api_key": SecretStr(api_key),
                "workspace_id": self.nams_workspace_id or os.environ.get("NAMS_WORKSPACE_ID"),
            }
            if self.nams_endpoint:
                fields["endpoint"] = self.nams_endpoint
            return MemorySettings(backend="nams", nams=NamsConfig(**fields))
        return MemorySettings(
            backend="bolt",
            neo4j=Neo4jConfig(
                uri=self.uri.resolve_value(),
                username=self.username.resolve_value() or "neo4j",
                password=SecretStr(self.password.resolve_value() or ""),
                database=self.database_name,
            ),
            llm=self.llm_model,
            embedding=self.embedding_model,
            extraction=ExtractionConfig(**self._extraction_fields()),
            resolution=ResolutionConfig(**(self.resolution or {})),
            memory=MemoryConfig(**(self.memory or {})),
        )

    @property
    def database_name(self) -> str:
        """The database the store connects to: ``database``, else ``NEO4J_DATABASE``, else ``"neo4j"``."""
        return self.database or os.environ.get("NEO4J_DATABASE") or "neo4j"

    def _explain(self, error: Exception) -> Exception:
        """Turn Neo4j's permission errors on first connect into a message that says what to change."""
        text = str(error)
        if "ACCESS on database" in text:
            msg = (
                f"Neo4j refused access to database {self.database_name!r} for this user. Set the database your user "
                "can use with Neo4jMemoryStore(database=...) or the NEO4J_DATABASE environment variable. "
                f"Neo4j said: {text}"
            )
            return PermissionError(msg)
        if "IndexAlreadyExists" in text or "conflicts with existing index" in text:
            msg = (
                f"Database {self.database_name!r} already has an index where neo4j-agent-memory needs a uniqueness "
                "constraint, usually because another application or memory system set up its own schema there. "
                "Use an empty database for agent memory (Neo4jMemoryStore(database=...) or NEO4J_DATABASE), or, if "
                "nothing uses the conflicting indexes, drop them and connect again. List them with: SHOW INDEXES "
                "YIELD name, labelsOrTypes, properties, owningConstraint WHERE owningConstraint IS NULL. "
                f"Neo4j said: {text}"
            )
            return RuntimeError(msg)
        if "Forbidden" in text or "not allowed" in text:
            msg = (
                "On first connect, neo4j-agent-memory creates the constraints and indexes it needs, and Neo4j refused "
                f"that for this user on database {self.database_name!r}. Use a user with schema (index and "
                f"constraint) privileges, or connect once as one to create them. Neo4j said: {text}"
            )
            return PermissionError(msg)
        return error

    def _create_client(self) -> Any:
        """
        Create the ``MemoryClient``, with an embedder that embeds identical texts once.

        Override in tests to inject a fake embedder or extractor.
        """
        settings = self._settings()
        if self.is_nams:  # NAMS embeds and extracts server-side
            return MemoryClient(settings)
        return MemoryClient(settings, embedder=_SharedEmbedder(adapt_to_legacy_embedder(settings.embedding)))

    async def _connected(self) -> Any:
        """The connected client. Runs on the store's loop; connects once (this creates schema and indexes)."""
        if self._client is not None:
            return self._client
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        async with self._connect_lock:
            if self._client is None:
                std_logging.getLogger("neo4j.notifications").setLevel(std_logging.ERROR)
                client = self._create_client()
                try:
                    await client.connect()
                except Exception as error:
                    try:
                        await client.close()
                    except Exception:  # noqa: S110 - best effort; the connect error is the one to report
                        pass
                    explained = self._explain(error)
                    if explained is error:
                        raise
                    raise explained from error
                self._client = client
        return self._client

    async def _disconnect(self) -> None:
        client, self._client = self._client, None
        self._connect_lock = None
        if client is not None:
            await client.close()

    def warm_up(self) -> None:
        """Connect to Neo4j. Idempotent and thread-safe."""
        self._runner.run(self._connected())

    async def warm_up_async(self) -> None:
        await self._runner.arun(self._connected())

    def close(self) -> None:
        """Close the connection and stop the background loop. The next call reconnects."""
        try:
            self._runner.run(self._disconnect())
        finally:
            self._runner.stop()

    async def close_async(self) -> None:
        try:
            await self._runner.arun(self._disconnect())
        finally:
            self._runner.stop()

    @property
    def client(self) -> Any:
        """
        The connected ``neo4j-agent-memory`` ``MemoryClient``, for anything the components don't cover.

        Its methods are coroutines bound to the store's loop: call them through ``store.run(...)`` or
        ``await store.arun(...)``, never directly on your own loop.
        """
        return self._runner.run(self._connected())

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine on the store's loop from synchronous code."""
        return self._runner.run(coro)

    async def arun(self, coro: Coroutine[Any, Any, T]) -> T:
        """Await a coroutine on the store's loop from any other event loop."""
        return await self._runner.arun(coro)

    # ------------------------------------------------------------------ sessions

    def create_session(self, *, user_id: str | None = None, metadata: dict[str, Any] | None = None) -> str:
        """
        Create a session and return its id, to pass as ``session_id`` to the components.

        With NAMS this is required: NAMS assigns conversation ids itself. With bolt any string works as a session
        id; this returns a new random one.
        """
        if not self.is_nams:
            return str(uuid.uuid4())

        async def _create() -> str:
            client = await self._connected()
            conversation = await client.short_term.create_conversation(
                str(uuid.uuid4()), user_identifier=user_id, metadata=metadata
            )
            return str(conversation.id)

        return self.run(_create())

    def _skipped_on_nams(self, method: str, what: str) -> bool:
        """On NAMS, memory types it cannot store are skipped with a warning instead of failing."""
        if self.is_nams:
            logger.warning(
                "{method}: skipped, because backend='nams' does not store {what}. Use backend='bolt' for them.",
                method=method,
                what=what,
            )
        return self.is_nams

    # ------------------------------------------------------------------ save memory directly
    #
    # The memory writer saves all three memories after each chat turn. These methods save one kind of memory
    # directly, for example to import history, seed preferences, or record a task that ran outside a pipeline.
    # Unlike the components, they always raise on errors: an explicit save that silently failed would be worse.

    def add_messages(
        self,
        session_id: str,
        messages: list[ChatMessage],
        *,
        extraction_policy: ExtractionPolicy | None = None,
        mentions: list[dict[str, Any]] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Short-term memory: save user and assistant messages to a session, in order.

        Entities are linked under ``extraction_policy`` (default: ``auto`` for user messages, ``tool_results``
        for assistant messages). System and tool messages are not stored as conversation messages; tool calls
        belong in reasoning memory (``add_trace``).

        :param session_id: The conversation session.
        :param messages: ``ChatMessage.from_user(...)`` / ``ChatMessage.from_assistant(...)`` messages.
        :param extraction_policy: Who decides what becomes an entity. See ``ExtractionPolicy``.
        :param mentions: Entities to link where the policy mode is ``"explicit"``: ``{"name", "type"}`` or ``{"id"}``.
        :param user_id: Optional user, for per-user scoping.
        :returns: ``{"messages_written": int, "entities": [...]}``.
        """
        _long_term.validate_mentions(mentions)
        policy = extraction_policy or ExtractionPolicy()

        async def _save() -> dict[str, Any]:
            client = await self._connected()
            written, entities = 0, []
            for message in messages:
                role = (
                    "user"
                    if message.is_from(ChatRole.USER)
                    else "assistant"
                    if message.is_from(ChatRole.ASSISTANT)
                    else None
                )
                if role is None or not message.text:
                    continue
                _, linked = await _long_term.write_with_policy(
                    client,
                    session_id=session_id,
                    role=role,
                    text=message.text,
                    mode=policy.user if role == "user" else policy.assistant,
                    policy=policy,
                    mentions=mentions,
                    turn=messages,
                    user_id=user_id,
                    strict=True,
                )
                written += 1
                entities.extend(linked)
            return {"messages_written": written, "entities": entities}

        return self.run(_save())

    def add_preference(
        self,
        category: str,
        preference: str,
        *,
        context: str | None = None,
        confidence: float = 1.0,
        user_id: str | None = None,
    ) -> str | None:
        """
        Long-term memory: save a user preference, e.g. ``add_preference("region", "Focus on companies in Germany")``.

        :returns: The preference id (a near-identical preference in the same category is reused), or ``None`` on
            NAMS, which does not store preferences: the call is skipped with a warning.
        """
        if self._skipped_on_nams("add_preference", "preferences"):
            return None
        item = {"category": category, "preference": preference, "context": context, "confidence": confidence}
        _long_term.validate_preferences([item])

        async def _save() -> str:
            client = await self._connected()
            saved = await client.long_term.add_preference(
                category, preference, context=context, confidence=confidence, user_identifier=user_id
            )
            return str(saved.id)

        return self.run(_save())

    def add_fact(self, subject: str, predicate: str, obj: str, *, confidence: float = 1.0) -> str | None:
        """
        Long-term memory: save a fact, e.g. ``add_fact("Pravesh", "works_at", "Northwind")``.

        :returns: The fact id, or ``None`` on NAMS, which does not store facts: the call is skipped with a warning.
        """
        if self._skipped_on_nams("add_fact", "facts"):
            return None
        _long_term.validate_facts([{"subject": subject, "predicate": predicate, "object": obj}])

        async def _save() -> str:
            client = await self._connected()
            saved = await client.long_term.add_fact(subject, predicate, obj, confidence=confidence)
            return str(saved.id)

        return self.run(_save())

    def add_entity(
        self,
        name: str,
        entity_type: str,
        *,
        description: str | None = None,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Long-term memory: save an entity, e.g. ``add_entity("Northwind Capital", "ORGANIZATION")``.

        Types follow the POLE+O model: ``PERSON``, ``ORGANIZATION``, ``LOCATION``, ``EVENT``, ``OBJECT``. The library
        resolves the name against existing entities, so "Northwind" may land on a stored "Northwind Capital".

        :returns: ``{"id", "name", "type", "resolution", "matched"}``. ``resolution`` is the library's decision:
            ``"none"`` (stored as new), ``"merged"`` (matched an existing entity, whose id is returned) or
            ``"flagged"`` (stored as new but marked as a possible duplicate of ``matched``; review these with
            ``find_potential_duplicates``).
        """
        if not name or not name.strip() or not entity_type or not entity_type.strip():
            msg = f"add_entity needs a name and a type, got name={name!r}, entity_type={entity_type!r}."
            raise ValueError(msg)

        async def _save() -> dict[str, Any]:
            client = await self._connected()
            result = await client.long_term.add_entity(
                name.strip(),
                entity_type.strip().upper(),
                description=description,
                aliases=aliases,
                attributes=attributes,
            )
            # bolt returns (entity, deduplication result); NAMS returns the entity and resolves server-side.
            entity, dedup = result if isinstance(result, tuple) else (result, None)
            return {
                "id": str(entity.id),
                "name": entity.name,
                "type": str(getattr(entity.type, "value", entity.type)),
                "resolution": str(getattr(dedup, "action", None) or "none"),
                "matched": getattr(dedup, "matched_entity_name", None),
            }

        return self.run(_save())

    def add_trace(
        self,
        session_id: str,
        task: str,
        messages: list[ChatMessage] | None = None,
        *,
        success: bool | None = None,
        user_id: str | None = None,
    ) -> str:
        """
        Reasoning memory: save a task with its tool calls, results and outcome, failed tasks included.

        :param session_id: The session the task belongs to.
        :param task: What was asked, e.g. the user's query.
        :param messages: The task's messages: assistant tool calls, ``ChatMessage.from_tool`` results and the final
            answer. If they include the user's message, only what follows the last user message is used.
        :param success: Whether the task succeeded. If ``None``: an answer with text and no tool errors.
        :param user_id: Optional user, for per-user scoping.
        :returns: The trace id. Failed traces store why they failed (``tool_error``, ``no_answer``,
            ``marked_failed``) and are recalled by the memory retriever unless ``include_failed_traces=False``.
        """
        if not task or not task.strip():
            msg = "add_trace needs a task."
            raise ValueError(msg)
        turn, answer = split_turn(messages, None)

        async def _save() -> str:
            client = await self._connected()
            return await _reasoning.record_trace(
                client,
                session_id=session_id,
                task=task,
                turn=turn,
                answer=answer,
                success=success if success is not None else _reasoning.default_success(turn, answer),
                user_id=user_id,
            )

        return self.run(_save())

    # ------------------------------------------------------------------ maintenance

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session's conversation and messages.

        Long-term memory (entities, preferences, facts) outlives sessions by design and is kept.
        """

        async def _delete() -> None:
            client = await self._connected()
            await client.short_term.clear_session(session_id)

        self.run(_delete())

    def find_potential_duplicates(self, limit: int = 20) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
        """
        Entity pairs the resolver flagged as possible duplicates, with their similarity.

        Empty on NAMS (with a warning): NAMS resolves duplicates itself; review them in the NAMS console.
        """
        if self._skipped_on_nams("find_potential_duplicates", "duplicate flags"):
            return []

        async def _find() -> list[Any]:
            client = await self._connected()
            return await client.long_term.find_potential_duplicates(limit=limit)

        def entity(e: Any) -> dict[str, Any]:
            return {"id": str(e.id), "name": e.name, "type": str(getattr(e.type, "value", e.type))}

        return [(entity(a), entity(b), float(score)) for a, b, score in self.run(_find())]

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store. Credentials are stored as env-var references only."""
        return default_to_dict(
            self,
            uri=self.uri.to_dict(),
            username=self.username.to_dict(),
            password=self.password.to_dict(),
            database=self.database,
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            extraction=self.extraction,
            resolution=self.resolution,
            memory=self.memory,
            backend=self.backend,
            nams_api_key=self.nams_api_key.to_dict(),
            nams_workspace_id=self.nams_workspace_id,
            nams_endpoint=self.nams_endpoint,
            nams=self.nams,
            strict=self.strict,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Neo4jMemoryStore:
        """Deserialize the store, reusing the live instance for an identical config."""
        key = json.dumps(data, sort_keys=True, default=str)
        existing = _REGISTRY.get(key)
        if existing is not None:
            return existing
        data = copy.deepcopy(data)
        deserialize_secrets_inplace(data["init_parameters"], keys=["uri", "username", "password", "nams_api_key"])
        store = default_from_dict(cls, data)
        _REGISTRY[key] = store
        return store
