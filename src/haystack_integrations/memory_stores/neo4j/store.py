"""Neo4jMemoryStore: connection, configuration and serialization shared by the memory components."""

from __future__ import annotations

import asyncio
import copy
import json
import logging as std_logging
import os
import weakref
from collections.abc import Coroutine
from typing import Any, TypeVar

from haystack import default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.memory_stores.neo4j._embedding import _SharedEmbedder
from haystack_integrations.memory_stores.neo4j._loop import _LoopRunner

_INSTALL_HINT = "Run 'pip install \"haystack-neo4j[memory]\"' to use the Neo4j agent memory components"

with LazyImport(message=_INSTALL_HINT) as memory_import:
    from neo4j_agent_memory import (
        ExtractionConfig,
        MemoryClient,
        MemoryConfig,
        MemorySettings,
        Neo4jConfig,
        ResolutionConfig,
    )
    from neo4j_agent_memory.embeddings.base import adapt_to_legacy_embedder
    from pydantic import SecretStr

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Pipeline YAML serializes the store once per component. Loading returns the live store for an identical
# config, so a loaded pipeline shares one connection. Keys hold env-var names only, never secret values.
_REGISTRY: weakref.WeakValueDictionary[str, Neo4jMemoryStore] = weakref.WeakValueDictionary()


class Neo4jMemoryStore:
    """
    Neo4j-backed short-term, long-term and reasoning memory, built on ``neo4j-agent-memory``.

    Holds the connection and configuration. ``Neo4jMemoryRetriever`` and ``Neo4jMemoryWriter`` both take a
    store, so they share one connection. No network call is made until ``warm_up()`` or first use.

    Usage example:

    ```python
    from haystack_integrations.memory_stores.neo4j import Neo4jMemoryStore

    store = Neo4jMemoryStore()  # reads NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        uri: Secret = Secret.from_env_var("NEO4J_URI"),  # noqa: B008
        username: Secret = Secret.from_env_var("NEO4J_USERNAME", strict=False),  # noqa: B008
        password: Secret = Secret.from_env_var("NEO4J_PASSWORD"),  # noqa: B008
        database: str | None = None,
        llm_model: str = "openai/gpt-5.4-mini",
        embedding_model: str = "openai/text-embedding-3-small",
        extraction: dict[str, Any] | None = None,
        resolution: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> None:
        """
        Create the store. Settings are validated here; nothing connects until first use.

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
        :param strict: If ``False`` (default), memory failures are logged and pipelines keep running.
            If ``True``, they raise.
        """
        memory_import.check()
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.extraction = extraction
        self.resolution = resolution
        self.memory = memory
        self.strict = strict

        # Validate the library configs now, so a typo fails at construction and not mid-pipeline.
        ExtractionConfig(**self._extraction_fields())
        ResolutionConfig(**(resolution or {}))
        MemoryConfig(**(memory or {}))

        self._runner = _LoopRunner()
        self._client: Any | None = None
        self._connect_lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------ settings and connection

    def _extraction_fields(self) -> dict[str, Any]:
        return dict(self.extraction) if self.extraction is not None else {"extractor_type": "llm"}

    def _settings(self) -> MemorySettings:
        """Build settings explicitly, so behaviour never depends on NAM_* variables or a local .env file."""
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
        """Entity pairs the resolver flagged as possible duplicates, with their similarity."""

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
        deserialize_secrets_inplace(data["init_parameters"], keys=["uri", "username", "password"])
        store = default_from_dict(cls, data)
        _REGISTRY[key] = store
        return store
