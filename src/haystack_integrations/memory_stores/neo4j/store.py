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
    from neo4j_agent_memory import MemoryClient, MemorySettings
    from neo4j_agent_memory.embeddings.base import adapt_to_legacy_embedder
    from pydantic import SecretStr

logger = logging.getLogger(__name__)
T = TypeVar("T")

Backend = Literal["bolt", "nams"]

# Secret fields of MemorySettings, and the store parameter that supplies each one instead.
_SECRET_FIELDS: dict[tuple[str, str], str | None] = {
    ("neo4j", "password"): "password",
    ("nams", "api_key"): "nams_api_key",
    ("nams", "headers"): None,
    ("geocoding", "api_key"): None,
    ("enrichment", "diffbot_api_key"): None,
}
# The library defaults to the "pipeline" extractor (spaCy, then GLiNER, then an LLM), which downloads models.
# Without an `extraction` section, the store uses the LLM extractor instead, which needs no download.
_DEFAULT_EXTRACTION = {"extractor_type": "llm"}
# The library's default embedding model, as a provider string (its default object form is deprecated).
_DEFAULT_EMBEDDING = "openai/text-embedding-3-small"

# Pipeline YAML serializes the store once per component. Loading returns the live store for an identical
# config, so a loaded pipeline shares one connection. Keys hold env-var names only, never secret values.
_REGISTRY: weakref.WeakValueDictionary[str, Neo4jMemoryStore] = weakref.WeakValueDictionary()


def _provider_string(value: Any) -> Any:
    """The library turns "provider/model" strings into provider objects; recover the string where it can."""
    if value is None or isinstance(value, (str, dict)) or hasattr(value, "model_dump"):
        return value
    model = getattr(value, "model", None)
    return model if isinstance(model, str) and "/" in model else value


def _settings_to_dict(settings: Any) -> dict[str, Any]:
    """The settings the user set, as a dict: ``MemorySettings`` keeps only explicitly set fields."""
    if settings is None:
        return {}
    if isinstance(settings, MemorySettings):
        # The library's validators mark some sections as set with empty defaults (e.g. llm={}): drop those.
        data = {k: v for k, v in settings.model_dump(exclude_unset=True).items() if v != {}}
        for key in ("embedding", "llm"):
            if key in data:
                data[key] = _provider_string(data[key])
        return data
    if isinstance(settings, dict):
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in settings.items()}
    msg = f"settings must be a neo4j_agent_memory.MemorySettings or a dict, got {type(settings).__name__}."
    raise TypeError(msg)


def _is_set(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "get_secret_value"):
        return bool(value.get_secret_value())
    return bool(value)


def _settings_to_json(settings: dict[str, Any]) -> dict[str, Any]:
    """Settings for YAML: secret values are refused, provider objects become their provider strings."""
    for (section, field), param in _SECRET_FIELDS.items():
        if _is_set((settings.get(section) or {}).get(field)):
            where = f"settings[{section!r}][{field!r}]"
            fix = (
                f"pass it as {param}=Secret.from_env_var(...) and leave it out of settings"
                if param
                else "set it through the library's NAM_* environment variables instead"
            )
            msg = f"{where} holds a secret, which would be written to YAML in plain text. To save this store, {fix}."
            raise ValueError(msg)

    data = dict(settings)
    for key in ("embedding", "llm"):
        if key not in data:
            continue
        value = data[key] = _provider_string(data[key])
        if value is not None and not isinstance(value, (str, dict)) and not hasattr(value, "model_dump"):
            msg = f"settings[{key!r}] is a {type(value).__name__} object that cannot be saved; use a provider string."
            raise ValueError(msg)

    def default(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", exclude_unset=True)
        if hasattr(value, "value"):  # enums
            return value.value
        msg = f"Cannot save a {type(value).__name__} in settings."
        raise ValueError(msg)

    return json.loads(json.dumps(data, default=default))


class Neo4jMemoryStore:
    """
    Neo4j-backed short-term, long-term and reasoning memory, built on ``neo4j-agent-memory``.

    Configured like the library's ``MemoryClient``: pass a ``MemorySettings`` (or a dict of the same shape) and,
    optionally, an ``embedder`` and ``extractor``. ``Neo4jMemoryRetriever`` and ``Neo4jMemoryWriter`` both take a
    store, so they share one connection. No network call is made until ``warm_up()`` or first use.

    Credentials stay out of ``settings``: they come from ``Secret`` parameters (``password``, ``nams_api_key``), so
    pipelines save to YAML with environment-variable names only. Anything ``settings`` leaves unset is filled from
    those parameters and the environment (``NEO4J_URI``, ``NEO4J_USERNAME``, ``NEO4J_DATABASE``,
    ``NAMS_WORKSPACE_ID``).

    Two backends, as in ``neo4j-agent-memory``:

    - ``"bolt"`` (default): the library connects to your Neo4j database directly. All memory types.
    - ``"nams"``: the library calls the hosted Neo4j Agent Memory Service (NAMS), which stores memory in your
      workspace and extracts entities server-side. NAMS assigns session (conversation) ids: get one with
      ``create_session()``. NAMS stores no preferences, facts or relations, and has no similar-task recall.

    Usage example:

    ```python
    from neo4j_agent_memory import ExtractionConfig, MemorySettings, NamsConfig
    from haystack_integrations.memory_stores.neo4j import Neo4jMemoryStore

    # bolt: connection from NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE
    store = Neo4jMemoryStore(MemorySettings(extraction=ExtractionConfig(extractor_type="spacy")))

    # NAMS: API key from NAMS_API_KEY
    nams_store = Neo4jMemoryStore(MemorySettings(backend="nams", nams=NamsConfig(workspace_id="ws-123")))
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        settings: MemorySettings | dict[str, Any] | None = None,
        *,
        uri: Secret = Secret.from_env_var("NEO4J_URI", strict=False),  # noqa: B008
        username: Secret = Secret.from_env_var("NEO4J_USERNAME", strict=False),  # noqa: B008
        password: Secret = Secret.from_env_var("NEO4J_PASSWORD", strict=False),  # noqa: B008
        nams_api_key: Secret = Secret.from_env_var("NAMS_API_KEY", strict=False),  # noqa: B008
        embedder: Any | None = None,
        extractor: Any | None = None,
        strict: bool = False,
    ) -> None:
        """
        Create the store. Settings are validated here; nothing connects until first use.

        :param settings: ``neo4j_agent_memory.MemorySettings``, or a dict with the same sections: ``backend``,
            ``neo4j``, ``nams``, ``embedding``, ``llm``, ``extraction``, ``resolution``, ``memory``, ``search``,
            ``schema_config``, ``geocoding``, ``enrichment``. Only what you set is kept. Without an ``extraction``
            section the LLM extractor is used (the library's own default, ``"pipeline"``, downloads models).
            Keep secrets out: use the parameters below.
        :param uri: bolt: Neo4j URI as a ``Secret``, used when ``settings`` sets no ``neo4j.uri``.
        :param username: bolt: Neo4j username as a ``Secret``, used when ``settings`` sets none. Default ``neo4j``.
        :param password: bolt: Neo4j password as a ``Secret``.
        :param nams_api_key: NAMS: API key as a ``Secret``.
        :param embedder: Optional embedder object, as for ``MemoryClient(embedder=...)``. A store with one runs, but
            cannot be saved to YAML; prefer a provider string in ``settings["embedding"]``.
        :param extractor: Optional entity extractor object, as for ``MemoryClient(extractor=...)``. Same caveat.
        :param strict: If ``False`` (default), memory failures are logged and pipelines keep running.
            If ``True``, they raise.
        """
        memory_import.check()
        self.settings = _settings_to_dict(settings)
        if self.backend not in get_args(Backend):
            msg = f"settings['backend'] must be one of {list(get_args(Backend))}, got {self.backend!r}."
            raise ValueError(msg)
        self.uri = uri
        self.username = username
        self.password = password
        self.nams_api_key = nams_api_key
        self.embedder = embedder
        self.extractor = extractor
        self.strict = strict
        if self.is_nams and (embedder is not None or extractor is not None):
            logger.warning("NAMS embeds and extracts server-side; the embedder and extractor objects are not used.")

        # Validate now, so a typo fails at construction and not mid-pipeline. Secrets are not read here.
        MemorySettings(**self._effective_settings(resolve_secrets=False))

        self._runner = _LoopRunner()
        self._client: Any | None = None
        self._connect_lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------ settings and connection

    @property
    def backend(self) -> str:
        return self.settings.get("backend") or "bolt"

    @property
    def is_nams(self) -> bool:
        return self.backend == "nams"

    @property
    def database_name(self) -> str:
        """The database: ``settings["neo4j"]["database"]``, else ``NEO4J_DATABASE``, else ``"neo4j"``."""
        configured = (self.settings.get("neo4j") or {}).get("database")
        return configured or os.environ.get("NEO4J_DATABASE") or "neo4j"

    def _effective_settings(self, *, resolve_secrets: bool) -> dict[str, Any]:
        """``settings``, with what they leave unset filled from the store's parameters and the environment."""
        data = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.settings.items()}
        data["backend"] = self.backend
        if self.is_nams:
            nams = dict(data.get("nams") or {})
            if not _is_set(nams.get("api_key")) and resolve_secrets:
                api_key = self.nams_api_key.resolve_value()
                if not api_key:
                    msg = (
                        "backend 'nams' needs an API key: set NAMS_API_KEY or pass "
                        "nams_api_key=Secret.from_env_var(...)."
                    )
                    raise ValueError(msg)
                nams["api_key"] = SecretStr(api_key)
            if not nams.get("workspace_id") and os.environ.get("NAMS_WORKSPACE_ID"):
                nams["workspace_id"] = os.environ["NAMS_WORKSPACE_ID"]
            data["nams"] = nams
        else:
            # Client-side defaults; NAMS embeds and extracts server-side and would warn about them.
            data.setdefault("extraction", dict(_DEFAULT_EXTRACTION))
            data.setdefault("embedding", _DEFAULT_EMBEDDING)
            neo4j = dict(data.get("neo4j") or {})
            if not neo4j.get("uri") and self.uri.resolve_value():
                neo4j["uri"] = self.uri.resolve_value()
            if not neo4j.get("username"):
                neo4j["username"] = self.username.resolve_value() or "neo4j"
            if not _is_set(neo4j.get("password")):
                neo4j["password"] = SecretStr((self.password.resolve_value() if resolve_secrets else None) or "")
            neo4j["database"] = self.database_name
            data["neo4j"] = neo4j
        return data

    def _settings(self) -> MemorySettings:
        return MemorySettings(**self._effective_settings(resolve_secrets=True))

    def _explain(self, error: Exception) -> Exception:
        """Turn Neo4j's permission errors on first connect into a message that says what to change."""
        text = str(error)
        set_database = 'settings={"neo4j": {"database": ...}} or the NEO4J_DATABASE environment variable'
        if "ACCESS on database" in text:
            msg = (
                f"Neo4j refused access to database {self.database_name!r} for this user. Set the database your user "
                f"can use with {set_database}. Neo4j said: {text}"
            )
            return PermissionError(msg)
        if "IndexAlreadyExists" in text or "conflicts with existing index" in text:
            msg = (
                f"Database {self.database_name!r} already has an index where neo4j-agent-memory needs a uniqueness "
                "constraint, usually because another application or memory system set up its own schema there. "
                f"Use an empty database for agent memory ({set_database}), or, if nothing uses the conflicting "
                "indexes, drop them and connect again. List them with: SHOW INDEXES "
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
        """Create the ``MemoryClient``. On bolt, its embedder embeds identical texts once (see ``_SharedEmbedder``)."""
        settings = self._settings()
        if self.is_nams:  # NAMS embeds and extracts server-side
            return MemoryClient(settings)
        kwargs: dict[str, Any] = {}
        try:
            embedder = adapt_to_legacy_embedder(self.embedder if self.embedder is not None else settings.embedding)
            kwargs["embedder"] = _SharedEmbedder(embedder)
        except TypeError:
            # A legacy EmbeddingConfig: the library builds that embedder itself (without the shared-embedding cache).
            pass
        if self.extractor is not None:
            kwargs["extractor"] = self.extractor
        return MemoryClient(settings, **kwargs)

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
        """
        Serialize the store: ``settings`` as plain data, credentials as environment-variable references only.

        Raises ``ValueError`` if ``settings`` holds a secret value, or if the store was given ``embedder`` or
        ``extractor`` objects, which cannot be saved.
        """
        if self.embedder is not None or self.extractor is not None:
            msg = (
                "A store with embedder or extractor objects cannot be saved to YAML. Configure them in settings "
                'instead, e.g. settings={"embedding": "openai/text-embedding-3-small"}.'
            )
            raise ValueError(msg)
        return default_to_dict(
            self,
            settings=_settings_to_json(self.settings),
            uri=self.uri.to_dict(),
            username=self.username.to_dict(),
            password=self.password.to_dict(),
            nams_api_key=self.nams_api_key.to_dict(),
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
