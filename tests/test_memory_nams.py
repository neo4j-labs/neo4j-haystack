"""
Tests for backend="nams" (hosted Neo4j Agent Memory Service), without a NAMS account.

The library's real NAMS client runs; only its HTTP transport is replaced by FakeNams, which records each request and
answers like the service. So these tests check which NAMS endpoints our components call and with what payloads.
"""

import asyncio
import json
import uuid
from typing import Any

import pytest

pytest.importorskip("neo4j_agent_memory")

from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage, ToolCall
from neo4j_agent_memory import NotFoundError
from neo4j_agent_memory.nams.transport import HttpTransport

from haystack_integrations.components.retrievers.neo4j import Neo4jMemoryRetriever
from haystack_integrations.components.writers.neo4j import Neo4jMemoryWriter
from haystack_integrations.memory_stores.neo4j import ExtractionPolicy, Neo4jMemoryStore, RecallConfig


class FakeNams:
    """In-memory stand-in for the NAMS REST API."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.conversations: dict[str, list[dict[str, Any]]] = {}
        self.entities: list[dict[str, Any]] = []

    def paths(self, method: str | None = None) -> list[str]:
        return [path for m, path, _ in self.calls if method is None or m == method]

    async def request(self, spec, *, path_params=None, json=None, params=None, **_):  # noqa: PLR0911
        method, template = spec.rest_method, spec.rest_path
        path = template.format(**(path_params or {}))
        self.calls.append((method, path, json))
        conv = (path_params or {}).get("conversation_id")

        if (method, template) == ("POST", "/conversations"):
            cid = str(uuid.uuid4())
            self.conversations[cid] = []
            return {"id": cid, "userId": json.get("userId"), "metadata": json.get("metadata") or {}}
        if template.startswith("/conversations/{conversation_id}") and conv not in self.conversations:
            raise NotFoundError(f"conversation {conv} not found")
        if (method, template) == ("GET", "/conversations/{conversation_id}"):
            return {"id": conv}
        if (method, template) == ("DELETE", "/conversations/{conversation_id}"):
            self.conversations.pop(conv)
            return None
        if (method, template) == ("GET", "/conversations/{conversation_id}/messages"):
            newest_first = list(reversed(self.conversations[conv]))
            return {"messages": newest_first[: (params or {}).get("limit", 100)]}
        if (method, template) == ("POST", "/conversations/{conversation_id}/messages"):
            message = {"id": str(uuid.uuid4()), "conversationId": conv, **json}
            self.conversations[conv].append(message)
            return message
        if (method, template) == ("POST", "/entities/search"):
            return {"entities": self.entities[: json.get("limit") or 10]}
        if (method, template) == ("POST", "/entities"):
            entity = {"id": str(uuid.uuid4()), "name": json["name"], "type": json["type"].lower()}
            self.entities.append(entity)
            return entity
        if (method, template) == ("POST", "/reasoning/steps"):
            return {"id": str(uuid.uuid4()), "reasoning": json.get("reasoning"), "actionTaken": json.get("actionTaken")}
        if (method, template) == ("POST", "/reasoning/tool-calls"):
            return {"id": str(uuid.uuid4()), "toolName": json["toolName"], "status": json.get("status")}
        msg = f"FakeNams has no answer for {method} {template}"
        raise AssertionError(msg)


@pytest.fixture
def nams(monkeypatch):
    fake = FakeNams()

    async def request(self, spec, **kwargs):
        return await fake.request(spec, **kwargs)

    monkeypatch.setattr(HttpTransport, "request", request)
    monkeypatch.setenv("NAMS_API_KEY", "nams-secret-key")
    monkeypatch.setenv("NAMS_WORKSPACE_ID", "ws-123")
    return fake


@pytest.fixture
def store(nams):
    s = Neo4jMemoryStore(backend="nams", nams={"validate_on_connect": False})
    yield s
    s.close()


@component
class EchoGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]):
        return {"replies": [ChatMessage.from_assistant("Echo: " + messages[-1].text)]}


# ---------------------------------------------------------------- configuration


class TestConfiguration:
    def test_settings_use_nams_config_from_env(self, store):
        settings = store._settings()
        assert settings.backend == "nams"
        assert settings.nams.api_key.get_secret_value() == "nams-secret-key"
        assert settings.nams.workspace_id == "ws-123"
        assert settings.nams.validate_on_connect is False

    def test_explicit_workspace_and_endpoint(self, nams):
        s = Neo4jMemoryStore(backend="nams", nams_workspace_id="ws-9", nams_endpoint="https://nams.example/v1")
        assert (s._settings().nams.workspace_id, s._settings().nams.endpoint) == ("ws-9", "https://nams.example/v1")

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("NAMS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="NAMS_API_KEY"):
            Neo4jMemoryStore(backend="nams")._settings()

    @pytest.mark.parametrize("kwargs", [{"backend": "http"}, {"nams": {"api_key": "x"}}, {"nams": {"headers": {}}}])
    def test_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            Neo4jMemoryStore(**kwargs)

    def test_invalid_nams_option(self):
        with pytest.raises(Exception, match="timeout"):  # pydantic ValidationError
            Neo4jMemoryStore(backend="nams", nams={"timeout": "soon"})

    def test_serialization_keeps_key_as_env_reference(self, store):
        data = store.to_dict()
        params = data["init_parameters"]
        assert params["backend"] == "nams"
        assert params["nams_api_key"] == {"type": "env_var", "env_vars": ["NAMS_API_KEY"], "strict": False}
        assert "nams-secret-key" not in json.dumps(data)
        restored = Neo4jMemoryStore.from_dict(data)
        assert restored.is_nams and restored.nams == {"validate_on_connect": False}

    def test_pipeline_yaml_round_trip(self, store):
        pipe = Pipeline()
        pipe.add_component("reader", Neo4jMemoryRetriever(store))
        pipe.add_component("writer", Neo4jMemoryWriter(store))
        loaded = Pipeline.loads(pipe.dumps())
        assert loaded.get_component("reader").store.is_nams
        assert loaded.get_component("reader").store is loaded.get_component("writer").store

    def test_bolt_is_still_the_default(self):
        assert Neo4jMemoryStore().backend == "bolt"


# ---------------------------------------------------------------- sessions


def test_create_session_returns_the_nams_conversation_id(store, nams):
    session_id = store.create_session(user_id="u1", metadata={"app": "demo"})
    assert session_id in nams.conversations
    assert nams.calls[-1] == ("POST", "/conversations", {"userId": "u1", "metadata": {"app": "demo"}})


def test_bolt_create_session_needs_no_connection(monkeypatch):
    s = Neo4jMemoryStore()
    s._connected = None  # would fail if called
    assert uuid.UUID(s.create_session())


def test_delete_session(store, nams):
    session_id = store.create_session()
    store.delete_session(session_id)
    assert ("DELETE", f"/conversations/{session_id}") in [(m, p) for m, p, _ in nams.calls]


# ---------------------------------------------------------------- writer


def test_writer_saves_messages_and_lets_nams_extract(store, nams):
    session_id = store.create_session()
    out = Neo4jMemoryWriter(store).run(
        query="I'm Pravesh", session_id=session_id, replies=[ChatMessage.from_assistant("Hi!")]
    )
    assert out["messages_written"] == 2
    assert out["entities"] == []  # extracted by NAMS in the background
    posted = [body for m, p, body in nams.calls if m == "POST" and p.endswith("/messages")]
    assert posted == [{"content": "I'm Pravesh", "role": "user"}, {"content": "Hi!", "role": "assistant"}]
    assert not any(p.startswith("/reasoning") for p in nams.paths())  # no tools: nothing to send for the trace


def test_writer_records_tool_steps_on_the_conversation(store, nams):
    session_id = store.create_session()
    ok, bad = ToolCall("search", {"country": "DE"}, id="c1"), ToolCall("lookup", {}, id="c2")
    transcript = [
        ChatMessage.from_user("Find companies"),
        ChatMessage.from_assistant("Searching", tool_calls=[ok, bad]),
        ChatMessage.from_tool('[{"name": "DeepL", "type": "ORGANIZATION"}]', origin=ok),
        ChatMessage.from_tool("HTTP 503", origin=bad, error=True),
        ChatMessage.from_assistant("DeepL."),
    ]
    out = Neo4jMemoryWriter(store).run(query="Find companies", session_id=session_id, messages=transcript)
    assert out["trace_id"] is not None
    steps = [body for m, p, body in nams.calls if p == "/reasoning/steps"]
    assert steps == [{"conversationId": session_id, "reasoning": "Searching", "actionTaken": "search, lookup"}]
    tool_calls = [body for m, p, body in nams.calls if p == "/reasoning/tool-calls"]
    assert [(c["toolName"], c["status"]) for c in tool_calls] == [("search", "success"), ("lookup", "error")]


@pytest.mark.parametrize(
    "inputs",
    [
        {"preferences": [{"category": "c", "preference": "p"}]},
        {"facts": [{"subject": "a", "predicate": "b", "object": "c"}]},
        {"mentions": [{"name": "Acme", "type": "ORGANIZATION"}]},
    ],
)
def test_writer_skips_inputs_nams_cannot_store(store, nams, inputs, caplog):
    session_id = store.create_session()
    out = Neo4jMemoryWriter(store).run(
        query="q", session_id=session_id, replies=[ChatMessage.from_assistant("a")], **inputs
    )
    assert out["messages_written"] == 2  # the turn itself is still saved
    assert "Ignoring writer inputs" in caplog.text
    assert not any(p in ("/preferences", "/facts") for p in nams.paths())


def test_writer_warns_that_extraction_policy_is_ignored(store, caplog):
    Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="review"))
    assert "ExtractionPolicy is ignored" in caplog.text


# ---------------------------------------------------------------- retriever


def test_retriever_reads_recent_history_and_entities_only(store, nams):
    session_id = store.create_session()
    writer = Neo4jMemoryWriter(store, record_reasoning="never")
    for i in range(3):
        writer.run(query=f"q{i}", session_id=session_id, replies=[ChatMessage.from_assistant(f"a{i}")])
    nams.entities.append({"id": str(uuid.uuid4()), "name": "Northwind Capital", "type": "organization"})

    recall = RecallConfig(history_limit=2)
    out = Neo4jMemoryRetriever(store, recall=recall, system_prompt="sys").run(query="next", session_id=session_id)
    assert [m.text for m in out["history"]] == ["q2", "a2"]
    assert [e["name"] for e in out["memories"]["entities"]] == ["Northwind Capital"]
    assert out["memories"]["preferences"] == out["memories"]["facts"] == out["memories"]["traces"] == []
    assert "- Northwind Capital (ORGANIZATION)" in out["context"]
    search = [body for m, p, body in nams.calls if p == "/entities/search"]
    assert search == [{"query": "next", "limit": 5}]


def test_retriever_unknown_session_has_no_history(store, nams):
    out = Neo4jMemoryRetriever(store).run(query="hi", session_id=str(uuid.uuid4()))
    assert out["history"] == [] and [m.text for m in out["messages"]] == ["hi"]


def test_pipeline_run_and_run_async(store, nams):
    session_id = store.create_session()
    pipe = Pipeline()
    pipe.add_component("reader", Neo4jMemoryRetriever(store))
    pipe.add_component("llm", EchoGenerator())
    pipe.add_component("writer", Neo4jMemoryWriter(store))
    pipe.connect("reader.messages", "llm.messages")
    pipe.connect("llm.replies", "writer.replies")

    assert pipe.run({"query": "hello", "session_id": session_id})["writer"]["messages_written"] == 2
    second = asyncio.run(pipe.run_async({"query": "again", "session_id": session_id}, include_outputs_from={"reader"}))
    assert [m.text for m in second["reader"]["history"]] == ["hello", "Echo: hello"]


# ---------------------------------------------------------------- store save methods


def test_store_save_methods(store, nams):
    session_id = store.create_session()
    assert store.add_messages(session_id, [ChatMessage.from_user("hi")])["messages_written"] == 1
    entity = store.add_entity("Northwind Capital", "organization")
    assert (entity["name"], entity["type"], entity["resolution"]) == ("Northwind Capital", "ORGANIZATION", "none")
    calls_before = len(nams.calls)
    assert store.add_preference("region", "Germany") is None
    assert store.add_fact("a", "b", "c") is None
    assert store.find_potential_duplicates() == []
    assert len(nams.calls) == calls_before  # skipped without calling NAMS
