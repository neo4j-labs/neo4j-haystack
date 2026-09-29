"""Unit tests for Neo4jGraphRAGRetriever. No network: the driver and neo4j-graphrag retrievers are mocked."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import neo4j
import neo4j.time
import pytest
from haystack import Document, Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.tools import ComponentTool
from haystack.utils import Secret
from neo4j_graphrag.types import RetrieverResult

from haystack_integrations.components.retrievers.neo4j_graphrag import Neo4jGraphRAGRetriever
from haystack_integrations.components.retrievers.neo4j_graphrag.retriever import _patch_cypher_version

MODULE = "haystack_integrations.components.retrievers.neo4j_graphrag.retriever"
RETRIEVAL_QUERY = "WITH node AS chunk, score RETURN chunk.text AS text, chunk.id AS chunk_id, score"
EMBEDDING = [0.1, 0.2, 0.3]


@pytest.fixture(autouse=True)
def neo4j_env(monkeypatch):
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "super-secret-password")


def make_retriever(search_type: str = "vector", **kwargs: Any) -> Neo4jGraphRAGRetriever:
    params: dict[str, Any] = {"url": "bolt://localhost:7687", "vector_index_name": "vec", "search_type": search_type}
    if search_type.startswith("hybrid"):
        params["fulltext_index_name"] = "ft"
    if search_type.endswith("_cypher"):
        params["retrieval_query"] = RETRIEVAL_QUERY
    params.update(kwargs)
    return Neo4jGraphRAGRetriever(**params)


class FakeEmbedder:
    """Duck-typed Haystack text embedder."""

    def __init__(self):
        self.warm_up_calls = 0
        self.run_calls: list[str] = []

    def warm_up(self):
        self.warm_up_calls += 1

    def run(self, text: str):
        self.run_calls.append(text)
        return {"embedding": EMBEDDING}


@pytest.fixture
def mocked_backend():
    """Mock the driver factory and all four neo4j-graphrag retriever classes."""
    with (
        patch(f"{MODULE}.neo4j.GraphDatabase.driver") as driver_factory,
        patch(f"{MODULE}.VectorRetriever") as vector,
        patch(f"{MODULE}.VectorCypherRetriever") as vector_cypher,
        patch(f"{MODULE}.HybridRetriever") as hybrid,
        patch(f"{MODULE}.HybridCypherRetriever") as hybrid_cypher,
    ):
        classes = {
            "vector": vector,
            "vector_cypher": vector_cypher,
            "hybrid": hybrid,
            "hybrid_cypher": hybrid_cypher,
        }
        for cls in classes.values():
            cls.return_value.search.return_value = RetrieverResult(items=[])
        yield driver_factory, classes


# ---------------------------------------------------------------- __init__ validation


class TestInit:
    @pytest.mark.parametrize("search_type", ["vector", "vector_cypher", "hybrid", "hybrid_cypher"])
    def test_valid_combinations(self, search_type):
        retriever = make_retriever(search_type)
        assert retriever.search_type == search_type
        assert retriever._driver is None  # no network in __init__

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"search_type": "keyword"}, "search_type"),
            ({"vector_index_name": ""}, "vector_index_name"),
            ({"url": ""}, "url"),
            ({"search_type": "hybrid", "fulltext_index_name": None}, "fulltext_index_name is required"),
            ({"search_type": "hybrid_cypher", "fulltext_index_name": None}, "fulltext_index_name is required"),
            ({"search_type": "vector", "fulltext_index_name": "ft"}, "fulltext_index_name is only valid"),
            ({"search_type": "vector_cypher", "fulltext_index_name": "ft"}, "fulltext_index_name is only valid"),
            ({"search_type": "vector_cypher", "retrieval_query": None}, "retrieval_query is required"),
            ({"search_type": "hybrid_cypher", "retrieval_query": "   "}, "retrieval_query is required"),
            ({"search_type": "vector", "retrieval_query": RETRIEVAL_QUERY}, "retrieval_query is only valid"),
            ({"search_type": "hybrid", "retrieval_query": RETRIEVAL_QUERY}, "retrieval_query is only valid"),
            ({"search_type": "vector_cypher", "return_properties": ["text"]}, "return_properties"),
            ({"search_type": "hybrid_cypher", "return_properties": ["text"]}, "return_properties"),
            ({"top_k": 0}, "top_k"),
            ({"top_k": -3}, "top_k"),
            ({"text_field": ""}, "text_field"),
            ({"text_embedder": object()}, "text_embedder"),
        ],
    )
    def test_invalid_combinations(self, kwargs, match):
        search_type = kwargs.pop("search_type", "vector")
        with pytest.raises(ValueError, match=match):
            make_retriever(search_type, **kwargs)

    def test_return_properties_allowed_for_plain_types(self):
        assert make_retriever("vector", return_properties=["text"]).return_properties == ["text"]
        assert make_retriever("hybrid", return_properties=["text"]).return_properties == ["text"]


# ---------------------------------------------------------------- serialization


class TestSerialization:
    def test_to_dict_has_env_secrets_and_no_password(self):
        data = make_retriever("hybrid_cypher", database="companies", cypher_version="25").to_dict()
        params = data["init_parameters"]
        assert data["type"] == f"{MODULE}.Neo4jGraphRAGRetriever"
        assert params["username"] == {"type": "env_var", "env_vars": ["NEO4J_USERNAME"], "strict": True}
        assert params["password"] == {"type": "env_var", "env_vars": ["NEO4J_PASSWORD"], "strict": True}
        assert "super-secret-password" not in json.dumps(data)
        assert params["retrieval_query"] == RETRIEVAL_QUERY
        assert params["text_embedder"] is None

    def test_token_secret_cannot_be_serialized(self):
        retriever = make_retriever(password=Secret.from_token("plain"))
        with pytest.raises(ValueError):
            retriever.to_dict()

    def test_round_trip_without_embedder(self):
        original = make_retriever("hybrid_cypher", database="companies", top_k=7, cypher_version="25")
        restored = Neo4jGraphRAGRetriever.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()
        assert isinstance(restored.password, Secret)
        assert restored.password.resolve_value() == "super-secret-password"
        assert restored.text_embedder is None

    def test_round_trip_with_embedder(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        embedder = OpenAITextEmbedder(model="text-embedding-3-small")
        original = make_retriever("vector", text_embedder=embedder)
        data = original.to_dict()
        assert data["init_parameters"]["text_embedder"]["type"].endswith("OpenAITextEmbedder")
        assert "fake-key" not in json.dumps(data)

        restored = Neo4jGraphRAGRetriever.from_dict(data)
        assert isinstance(restored.text_embedder, OpenAITextEmbedder)
        assert restored.text_embedder.model == "text-embedding-3-small"
        assert restored.to_dict() == original.to_dict()  # from_dict deserializes `data` in place

    def test_pipeline_dumps_loads_default_allowlist(self):
        pipeline = Pipeline()
        pipeline.add_component("retriever", make_retriever("vector_cypher", database="companies"))
        yaml_str = pipeline.dumps()
        assert "super-secret-password" not in yaml_str
        assert "haystack_integrations.components.retrievers.neo4j_graphrag" in yaml_str

        loaded = Pipeline.loads(yaml_str)
        retriever = loaded.get_component("retriever")
        assert isinstance(retriever, Neo4jGraphRAGRetriever)
        assert retriever.retrieval_query == RETRIEVAL_QUERY
        assert loaded.dumps() == yaml_str


# ---------------------------------------------------------------- warm_up / close


class TestLifecycle:
    @pytest.mark.parametrize("search_type", ["vector", "vector_cypher", "hybrid", "hybrid_cypher"])
    def test_warm_up_builds_matching_retriever(self, mocked_backend, search_type):
        driver_factory, classes = mocked_backend
        retriever = make_retriever(search_type, database="companies")
        retriever.warm_up()

        driver_factory.assert_called_once_with("bolt://localhost:7687", auth=("neo4j", "super-secret-password"))
        cls = classes[search_type]
        cls.assert_called_once()
        kwargs = cls.call_args.kwargs
        assert kwargs["driver"] is driver_factory.return_value
        assert kwargs["embedder"] is None
        assert kwargs["neo4j_database"] == "companies"
        assert kwargs["result_formatter"] == retriever._format_record
        for other, other_cls in classes.items():
            if other != search_type:
                other_cls.assert_not_called()

    def test_warm_up_is_idempotent(self, mocked_backend):
        driver_factory, classes = mocked_backend
        embedder = FakeEmbedder()
        retriever = make_retriever("vector", text_embedder=embedder)
        retriever.warm_up()
        retriever.warm_up()
        assert driver_factory.call_count == 1
        assert classes["vector"].call_count == 1
        assert embedder.warm_up_calls == 1

    def test_run_warms_up_lazily(self, mocked_backend):
        driver_factory, _ = mocked_backend
        retriever = make_retriever("vector")
        retriever.run(query="q", query_embedding=EMBEDDING)
        driver_factory.assert_called_once()

    def test_close_closes_driver_and_allows_reconnect(self, mocked_backend):
        driver_factory, _ = mocked_backend
        retriever = make_retriever("vector")
        retriever.warm_up()
        retriever.close()
        driver_factory.return_value.close.assert_called_once()
        assert retriever._driver is None
        retriever.warm_up()
        assert driver_factory.call_count == 2

    def test_driver_closed_if_retriever_build_fails(self, mocked_backend):
        driver_factory, classes = mocked_backend
        classes["vector"].side_effect = RuntimeError("index not found")
        retriever = make_retriever("vector")
        with pytest.raises(RuntimeError):
            retriever.warm_up()
        driver_factory.return_value.close.assert_called_once()


# ---------------------------------------------------------------- run()


class TestRun:
    @pytest.mark.parametrize("search_type", ["vector", "vector_cypher"])
    def test_vector_types_pass_vector_not_text(self, mocked_backend, search_type):
        _, classes = mocked_backend
        make_retriever(search_type).run(query="hello", query_embedding=EMBEDDING)
        kwargs = classes[search_type].return_value.search.call_args.kwargs
        assert kwargs["query_vector"] == EMBEDDING
        assert "query_text" not in kwargs
        assert kwargs["top_k"] == 5

    @pytest.mark.parametrize("search_type", ["hybrid", "hybrid_cypher"])
    def test_hybrid_types_pass_vector_and_text(self, mocked_backend, search_type):
        _, classes = mocked_backend
        make_retriever(search_type).run(query="hello", query_embedding=EMBEDDING, top_k=2)
        kwargs = classes[search_type].return_value.search.call_args.kwargs
        assert kwargs["query_vector"] == EMBEDDING
        assert kwargs["query_text"] == "hello"
        assert kwargs["top_k"] == 2

    @pytest.mark.parametrize("search_type", ["vector_cypher", "hybrid_cypher"])
    def test_query_params_forwarded_for_cypher(self, mocked_backend, search_type):
        _, classes = mocked_backend
        params = {"since": "2024-01-01"}
        make_retriever(search_type).run(query="q", query_embedding=EMBEDDING, query_params=params)
        assert classes[search_type].return_value.search.call_args.kwargs["query_params"] == params

    @pytest.mark.parametrize("search_type", ["vector", "hybrid"])
    def test_query_params_rejected_for_plain(self, mocked_backend, search_type):
        with pytest.raises(ValueError, match="query_params"):
            make_retriever(search_type).run(query="q", query_embedding=EMBEDDING, query_params={"since": "x"})

    def test_query_embedding_bypasses_embedder(self, mocked_backend):
        _, classes = mocked_backend
        embedder = FakeEmbedder()
        make_retriever("vector", text_embedder=embedder).run(query="q", query_embedding=[9.0, 9.0])
        assert embedder.run_calls == []
        assert classes["vector"].return_value.search.call_args.kwargs["query_vector"] == [9.0, 9.0]

    def test_embedder_used_without_query_embedding(self, mocked_backend):
        _, classes = mocked_backend
        embedder = FakeEmbedder()
        make_retriever("hybrid", text_embedder=embedder).run(query="what about AI?")
        assert embedder.run_calls == ["what about AI?"]
        assert embedder.warm_up_calls == 1
        assert classes["hybrid"].return_value.search.call_args.kwargs["query_vector"] == EMBEDDING

    def test_no_embedding_and_no_embedder_raises(self, mocked_backend):
        driver_factory, _ = mocked_backend
        with pytest.raises(ValueError, match="text_embedder"):
            make_retriever("vector").run(query="q")
        driver_factory.assert_not_called()

    def test_run_returns_documents_from_formatter(self, mocked_backend):
        _, classes = mocked_backend
        retriever = make_retriever("vector_cypher")
        record = {"text": "chunk text", "chunk_id": "c1", "score": 0.9}
        classes["vector_cypher"].return_value.search.return_value = RetrieverResult(
            items=[retriever._format_record(record)]
        )
        docs = retriever.run(query="q", query_embedding=EMBEDDING)["documents"]
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].content == "chunk text"
        assert docs[0].meta == {"chunk_id": "c1"}
        assert docs[0].score == 0.9


# ---------------------------------------------------------------- record conversion


class TestRecordConversion:
    def test_plain_record(self):
        record = {
            "node": {
                "text": "Neo4j raises funding",
                "id": "chunk-1",
                "index": 3,
                "embedding": [0.1] * 8,
                "embedding_sbert": [0.2] * 8,
                "embedding_google": [1, 2, 3],
                "embedding_indexed": None,  # neo4j-graphrag nulls the indexed embedding
                "tags": ["a", "b"],
            },
            "nodeLabels": ["Chunk"],
            "elementId": "4:abc:1",
            "id": "4:abc:1",
            "score": 0.87,
        }
        doc = make_retriever("vector")._record_to_document(record)
        assert doc.content == "Neo4j raises funding"
        assert doc.score == pytest.approx(0.87)
        assert doc.meta == {
            "id": "chunk-1",
            "index": 3,
            "tags": ["a", "b"],
            "labels": ["Chunk"],
            "element_id": "4:abc:1",
        }

    def test_plain_record_custom_text_field(self):
        record = {"node": {"summary": "S", "text": "T"}, "nodeLabels": ["Article"], "elementId": "e", "score": 1}
        doc = make_retriever("hybrid", text_field="summary")._record_to_document(record)
        assert doc.content == "S"
        assert doc.meta["text"] == "T"
        assert doc.score == 1.0

    def test_cypher_record(self):
        record = {
            "text": "chunk text",
            "article_id": "a1",
            "title": "Title",
            "companies": ["Neo4j", "Acme"],
            "score": 0.5,
        }
        doc = make_retriever("hybrid_cypher")._record_to_document(record)
        assert doc.content == "chunk text"
        assert doc.score == 0.5
        assert doc.meta == {"article_id": "a1", "title": "Title", "companies": ["Neo4j", "Acme"]}

    def test_neo4j_time_is_iso_string(self):
        dt = neo4j.time.DateTime(2024, 5, 1, 12, 30, 0)
        date = neo4j.time.Date(2023, 1, 2)
        record = {
            "text": "t",
            "date": dt,
            "nested": {"dates": [date], "when": dt},
            "score": 0.1,
        }
        doc = make_retriever("vector_cypher")._record_to_document(record)
        assert doc.meta["date"] == dt.iso_format()
        assert doc.meta["nested"] == {"dates": [date.iso_format()], "when": dt.iso_format()}
        json.dumps(doc.meta)  # meta is JSON-safe

    def test_neo4j_record_object(self):
        record = neo4j.Record({"text": "hello", "extra": 1, "score": 0.3})
        doc = make_retriever("vector_cypher")._record_to_document(record)
        assert doc.content == "hello"
        assert doc.meta == {"extra": 1}
        assert doc.score == pytest.approx(0.3)

    def test_missing_score(self):
        doc = make_retriever("vector_cypher")._record_to_document({"text": "x"})
        assert doc.score is None


# ---------------------------------------------------------------- cypher_version


class TestCypherVersion:
    def _driver_with_mock(self):
        driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("u", "p"))  # lazy, no connection
        inner = MagicMock(return_value="result")
        driver.execute_query = inner
        return driver, inner

    def test_prefixes_queries_on_instance_only(self):
        class_method_before = neo4j.Driver.execute_query
        driver, inner = self._driver_with_mock()
        try:
            _patch_cypher_version(driver, "25")
            assert driver.execute_query("MATCH (n) RETURN n", {"a": 1}, database_="db") == "result"
            inner.assert_called_once_with("CYPHER 25 MATCH (n) RETURN n", {"a": 1}, database_="db")

            assert isinstance(driver, neo4j.Driver)
            assert neo4j.Driver.execute_query is class_method_before

            other, other_inner = self._driver_with_mock()
            other.execute_query("RETURN 1")
            other_inner.assert_called_once_with("RETURN 1")
            other.close()
        finally:
            driver.close()

    @pytest.mark.parametrize("query", ["CYPHER 25 RETURN 1", "  cypher 5 RETURN 1"])
    def test_never_double_prefixes(self, query):
        driver, inner = self._driver_with_mock()
        try:
            _patch_cypher_version(driver, "25")
            driver.execute_query(query)
            inner.assert_called_once_with(query)
        finally:
            driver.close()

    def test_warm_up_patches_only_when_set(self, mocked_backend):
        driver_factory, _ = mocked_backend
        fake_driver = MagicMock()
        original = fake_driver.execute_query
        driver_factory.return_value = fake_driver

        make_retriever("vector").warm_up()
        assert fake_driver.execute_query is original

        make_retriever("vector", cypher_version="25").warm_up()
        assert fake_driver.execute_query is not original
        fake_driver.execute_query("RETURN 1")
        original.assert_called_once_with("CYPHER 25 RETURN 1")


# ---------------------------------------------------------------- ComponentTool


class TestComponentTool:
    def test_component_tool_builds(self):
        tool = ComponentTool(component=make_retriever("hybrid_cypher"), name="graph_search", description="Search")
        props = tool.parameters["properties"]
        assert "query" in props
        assert "retrieval_query" not in props
        assert tool.parameters["required"] == ["query"]

    def test_component_tool_parameters_limit_to_query(self, mocked_backend):
        _, classes = mocked_backend
        embedder = FakeEmbedder()
        tool = ComponentTool(
            component=make_retriever("vector", text_embedder=embedder),
            name="vector_search",
            description="Semantic search over news chunks",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search text"}},
                "required": ["query"],
            },
        )
        assert list(tool.parameters["properties"]) == ["query"]
        result = tool.invoke(query="AI investments")
        assert embedder.run_calls == ["AI investments"]
        assert "documents" in result
        classes["vector"].return_value.search.assert_called_once()
