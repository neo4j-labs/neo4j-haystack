"""Haystack retriever component that wraps Neo4j's official ``neo4j-graphrag`` retrievers."""

from __future__ import annotations

import logging
from typing import Any, Literal, get_args

import neo4j
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.core.serialization import component_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_component_inplace, deserialize_secrets_inplace

from haystack_integrations.common.neo4j.utils import is_number_list, patch_cypher_version, to_json_safe

_INSTALL_HINT = "Run 'pip install \"haystack-neo4j[graphrag]\"' to use Neo4jGraphRAGRetriever"

with LazyImport(message=_INSTALL_HINT) as graphrag_import:
    from neo4j_graphrag.retrievers import (
        HybridCypherRetriever,
        HybridRetriever,
        VectorCypherRetriever,
        VectorRetriever,
    )
    from neo4j_graphrag.types import RetrieverResultItem

logger = logging.getLogger(__name__)

SearchType = Literal["vector", "vector_cypher", "hybrid", "hybrid_cypher"]
SEARCH_TYPES: tuple[str, ...] = get_args(SearchType)

# Metadata key the result formatter uses to hand the converted Document back to ``run()``.
_DOCUMENT_KEY = "__haystack_document__"


@component
class Neo4jGraphRAGRetriever:
    """
    Retrieves Documents from an existing Neo4j graph using the official ``neo4j-graphrag`` retrievers.

    One component covers four search types:

    - ``vector``: vector similarity search on a vector index.
    - ``vector_cypher``: vector search followed by a graph traversal defined in ``retrieval_query``.
    - ``hybrid``: vector plus full-text search.
    - ``hybrid_cypher``: hybrid search followed by a graph traversal defined in ``retrieval_query``.

    Query embeddings come from any Haystack text embedder passed as ``text_embedder``, or from the
    ``query_embedding`` input of ``run()``.

    Usage example:

    ```python
    from haystack.utils import Secret
    from haystack_integrations.components.retrievers.neo4j import Neo4jGraphRAGRetriever

    retriever = Neo4jGraphRAGRetriever(
        url="neo4j+s://demo.neo4jlabs.com",
        username=Secret.from_env_var("NEO4J_USERNAME"),
        password=Secret.from_env_var("NEO4J_PASSWORD"),
        database="companies",
        search_type="hybrid",
        vector_index_name="news_sbert",
        fulltext_index_name="news_fulltext",
        text_embedder=my_haystack_text_embedder,
        cypher_version="25",
    )
    documents = retriever.run(query="Which companies are investing in AI?")["documents"]
    ```
    """

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        *,
        url: str,
        vector_index_name: str,
        username: Secret = Secret.from_env_var("NEO4J_USERNAME"),  # noqa: B008
        password: Secret = Secret.from_env_var("NEO4J_PASSWORD"),  # noqa: B008
        database: str | None = None,
        search_type: SearchType = "vector",
        fulltext_index_name: str | None = None,
        retrieval_query: str | None = None,
        return_properties: list[str] | None = None,
        text_field: str = "text",
        text_embedder: Any | None = None,
        top_k: int = 5,
        cypher_version: str | None = None,
    ) -> None:
        """
        Create a Neo4jGraphRAGRetriever. No network connection is made until ``warm_up()``.

        :param url: Neo4j connection URL, e.g. ``"bolt://localhost:7687"``.
        :param vector_index_name: Name of the existing vector index.
        :param username: Neo4j username as a ``Secret``.
        :param password: Neo4j password as a ``Secret``.
        :param database: Neo4j database name. Uses the server default if ``None``.
        :param search_type: One of ``"vector"``, ``"vector_cypher"``, ``"hybrid"``, ``"hybrid_cypher"``.
        :param fulltext_index_name: Name of the existing full-text index. Required for ``hybrid*``.
        :param retrieval_query: Cypher run after the index search, starting from ``node`` and ``score``.
            Required for ``*_cypher``. It is fixed configuration and is never a ``run()`` input.
        :param return_properties: Node properties to return. Only for ``vector`` and ``hybrid``.
        :param text_field: Node property (or ``retrieval_query`` column) that becomes ``Document.content``.
        :param text_embedder: A Haystack text embedder used to embed the query when ``run()`` gets no
            ``query_embedding``.
        :param top_k: Default number of results.
        :param cypher_version: If set (e.g. ``"25"``), prefixes the component's queries with
            ``CYPHER <version>``. Needed when the database defaults to an older Cypher version.
        """
        graphrag_import.check()

        if search_type not in SEARCH_TYPES:
            msg = f"search_type must be one of {list(SEARCH_TYPES)}, got {search_type!r}."
            raise ValueError(msg)
        if not url:
            msg = "url is required."
            raise ValueError(msg)
        if not vector_index_name:
            msg = "vector_index_name is required."
            raise ValueError(msg)

        is_hybrid = search_type.startswith("hybrid")
        is_cypher = search_type.endswith("_cypher")

        if is_hybrid and not fulltext_index_name:
            msg = f"fulltext_index_name is required for search_type={search_type!r}."
            raise ValueError(msg)
        if not is_hybrid and fulltext_index_name is not None:
            msg = f"fulltext_index_name is only valid for hybrid search types, not {search_type!r}."
            raise ValueError(msg)
        if is_cypher and not (retrieval_query and retrieval_query.strip()):
            msg = f"retrieval_query is required for search_type={search_type!r}."
            raise ValueError(msg)
        if not is_cypher and retrieval_query is not None:
            msg = (
                f"retrieval_query is only valid for 'vector_cypher' and 'hybrid_cypher', not {search_type!r}. "
                "Use search_type='vector_cypher' or 'hybrid_cypher' to run a graph traversal."
            )
            raise ValueError(msg)
        if is_cypher and return_properties is not None:
            msg = (
                f"return_properties is not supported for search_type={search_type!r}; "
                "choose the returned columns in retrieval_query instead."
            )
            raise ValueError(msg)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            msg = f"top_k must be a positive integer, got {top_k!r}."
            raise ValueError(msg)
        if not text_field:
            msg = "text_field must be a non-empty string."
            raise ValueError(msg)
        if text_embedder is not None and not callable(getattr(text_embedder, "run", None)):
            msg = "text_embedder must be a Haystack text embedder component with a run(text=...) method."
            raise ValueError(msg)

        self.url = url
        self.username = username
        self.password = password
        self.database = database
        self.search_type = search_type
        self.vector_index_name = vector_index_name
        self.fulltext_index_name = fulltext_index_name
        self.retrieval_query = retrieval_query
        self.return_properties = return_properties
        self.text_field = text_field
        self.text_embedder = text_embedder
        self.top_k = top_k
        self.cypher_version = cypher_version

        self._driver: neo4j.Driver | None = None
        self._retriever: Any | None = None
        self._embedder_warmed_up = False

    # ------------------------------------------------------------------ lifecycle

    def warm_up(self) -> None:
        """Warm up the embedder, connect to Neo4j and build the neo4j-graphrag retriever. Idempotent."""
        if self.text_embedder is not None and not self._embedder_warmed_up:
            if hasattr(self.text_embedder, "warm_up"):
                self.text_embedder.warm_up()
            self._embedder_warmed_up = True

        if self._retriever is not None:
            return

        driver = neo4j.GraphDatabase.driver(
            self.url, auth=(self.username.resolve_value(), self.password.resolve_value())
        )
        if self.cypher_version:
            patch_cypher_version(driver, self.cypher_version)

        try:
            self._retriever = self._build_retriever(driver)
        except Exception:
            driver.close()
            raise
        self._driver = driver

    def _build_retriever(self, driver: neo4j.Driver) -> Any:
        common: dict[str, Any] = {
            "driver": driver,
            "embedder": None,
            "result_formatter": self._format_record,
            "neo4j_database": self.database,
        }
        if self.search_type == "vector":
            return VectorRetriever(
                index_name=self.vector_index_name, return_properties=self.return_properties, **common
            )
        if self.search_type == "vector_cypher":
            return VectorCypherRetriever(
                index_name=self.vector_index_name, retrieval_query=self.retrieval_query, **common
            )
        if self.search_type == "hybrid":
            return HybridRetriever(
                vector_index_name=self.vector_index_name,
                fulltext_index_name=self.fulltext_index_name,
                return_properties=self.return_properties,
                **common,
            )
        return HybridCypherRetriever(
            vector_index_name=self.vector_index_name,
            fulltext_index_name=self.fulltext_index_name,
            retrieval_query=self.retrieval_query,
            **common,
        )

    def close(self) -> None:
        """Close the Neo4j driver. The next ``run()`` or ``warm_up()`` reconnects."""
        if self._driver is not None:
            self._driver.close()
        self._driver = None
        self._retriever = None

    # ------------------------------------------------------------------ run

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int | None = None,
        query_params: dict[str, Any] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve Documents from Neo4j.

        :param query: The query text. Embedded with ``text_embedder`` if ``query_embedding`` is not given,
            and used for full-text search in the hybrid search types.
        :param query_embedding: Precomputed query embedding. Skips the ``text_embedder``.
        :param top_k: Number of results. Defaults to the ``top_k`` set at init.
        :param query_params: Values for parameters used in ``retrieval_query`` (e.g. ``{"since": "2024-01-01"}``).
            Only valid for ``vector_cypher`` and ``hybrid_cypher``.
        :returns: A dictionary with ``documents``: the retrieved Documents, with graph context in ``meta``.
        """
        is_hybrid = self.search_type.startswith("hybrid")
        is_cypher = self.search_type.endswith("_cypher")

        if query_params and not is_cypher:
            msg = f"query_params are only supported for 'vector_cypher' and 'hybrid_cypher', not {self.search_type!r}."
            raise ValueError(msg)

        if query_embedding is None:
            if self.text_embedder is None:
                msg = "No query_embedding was given and no text_embedder is configured; provide one of them."
                raise ValueError(msg)
            self.warm_up()
            query_embedding = self.text_embedder.run(text=query)["embedding"]

        self.warm_up()

        search_kwargs: dict[str, Any] = {
            "query_vector": list(query_embedding),
            "top_k": top_k if top_k is not None else self.top_k,
        }
        if is_hybrid:
            search_kwargs["query_text"] = query
        if is_cypher and query_params:
            search_kwargs["query_params"] = query_params

        result = self._retriever.search(**search_kwargs)  # type: ignore[union-attr]
        documents = [self._item_to_document(item) for item in result.items]
        return {"documents": documents}

    def _format_record(self, record: Any) -> RetrieverResultItem:
        """``result_formatter`` for neo4j-graphrag: wraps the converted Document in a RetrieverResultItem."""
        document = self._record_to_document(record)
        return RetrieverResultItem(content=document.content or "", metadata={_DOCUMENT_KEY: document})

    @staticmethod
    def _item_to_document(item: RetrieverResultItem) -> Document:
        metadata = item.metadata or {}
        if _DOCUMENT_KEY in metadata:
            return metadata[_DOCUMENT_KEY]
        return Document(content=str(item.content), meta=to_json_safe({k: v for k, v in metadata.items()}))

    def _record_to_document(self, record: Any) -> Document:
        """Convert a Neo4j record (or a plain dict) into a Haystack Document."""
        data = dict(record.items())
        score = data.pop("score", None)
        score = float(score) if score is not None else None

        if self.search_type.endswith("_cypher"):
            content = data.pop(self.text_field, None)
            meta = data
        else:
            node = data.get("node")
            props = dict(node.items()) if node is not None and hasattr(node, "items") else {}
            # Drop embeddings. neo4j-graphrag nulls out the indexed one (e.g. `embedding_sbert: None`);
            # Neo4j properties are never null, so a None here is always such a placeholder.
            props = {k: v for k, v in props.items() if v is not None and not is_number_list(v)}
            content = props.pop(self.text_field, None)
            meta = props
            if data.get("nodeLabels") is not None:
                meta["labels"] = list(data["nodeLabels"])
            if data.get("elementId") is not None:
                meta["element_id"] = data["elementId"]
            if data.get("id") is not None and "id" not in meta:
                meta["id"] = data["id"]

        if content is not None and not isinstance(content, str):
            content = str(content)

        return Document(content=content, meta=to_json_safe(meta), score=score)

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary. Secrets are stored as env-var references only."""
        embedder = (
            component_to_dict(obj=self.text_embedder, name="text_embedder") if self.text_embedder is not None else None
        )
        return default_to_dict(
            self,
            url=self.url,
            username=self.username.to_dict(),
            password=self.password.to_dict(),
            database=self.database,
            search_type=self.search_type,
            vector_index_name=self.vector_index_name,
            fulltext_index_name=self.fulltext_index_name,
            retrieval_query=self.retrieval_query,
            return_properties=self.return_properties,
            text_field=self.text_field,
            text_embedder=embedder,
            top_k=self.top_k,
            cypher_version=self.cypher_version,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Neo4jGraphRAGRetriever:
        """Deserialize this component from a dictionary."""
        params = data.get("init_parameters", {})
        deserialize_secrets_inplace(params, keys=["username", "password"])
        if params.get("text_embedder") is not None:
            deserialize_component_inplace(params, key="text_embedder")
        return default_from_dict(cls, data)
