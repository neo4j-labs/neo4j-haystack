"""Tests for the agent memory tools (retrieve_memories / store_memory). No network: the MemoryClient is mocked."""

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("neo4j_agent_memory")

from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.memory_stores.neo4j import Neo4jMemoryStore, RecallConfig
from haystack_integrations.tools.neo4j import Neo4jMemoryRetrieverTool, Neo4jMemoryWriterTool

from .test_memory_unit import attach, fake_client


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "super-secret-password")


@pytest.fixture
def store():
    s = Neo4jMemoryStore()
    yield s
    s._runner.stop()


def preference(text: str, category: str = "region") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(), category=category, preference=text, context=None, confidence=1.0, metadata={"similarity": 0.9}
    )


# ---------------------------------------------------------------- retrieve_memories


class TestRetrieverTool:
    def test_schema_shows_only_query(self, store):
        tool = Neo4jMemoryRetrieverTool(memory_store=store)
        assert tool.name == "retrieve_memories"
        assert list(tool.parameters["properties"]) == ["query"]
        assert tool.inputs_from_state == {"session_id": "session_id", "user_id": "user_id"}

    def test_query_returns_long_term_context_without_history(self, store):
        client = attach(store, fake_client())
        client.long_term.search_preferences.return_value = [preference("Focus on Germany")]
        text = Neo4jMemoryRetrieverTool(memory_store=store).retrieve(
            "where do I invest?", session_id="s1", user_id="u1"
        )
        assert "- [region] Focus on Germany" in text
        client.query.cypher.assert_not_awaited()  # no conversation history for query recall

    def test_no_query_returns_recent_conversation_and_user_preferences(self, store):
        client = attach(store, fake_client())
        client.query.cypher.return_value = [{"id": "1", "role": "user", "content": "hi"}]
        client.long_term.get_preferences_for = AsyncMock(return_value=[preference("Focus on Germany")])
        text = Neo4jMemoryRetrieverTool(memory_store=store).retrieve(session_id="s1", user_id="u1")
        assert text == "Recent conversation:\n- user: hi\nUser preferences:\n- [region] Focus on Germany"
        client.long_term.get_preferences_for.assert_awaited_once_with("u1")

    def test_nothing_found(self, store):
        attach(store, fake_client())
        assert Neo4jMemoryRetrieverTool(memory_store=store).retrieve("anything", session_id="s") == "No memories found."

    def test_failure_is_reported_to_the_model(self, store):
        store._connected = AsyncMock(side_effect=ConnectionError("down"))
        assert (
            Neo4jMemoryRetrieverTool(memory_store=store).retrieve("x", session_id="s")
            == "Memory is unavailable right now."
        )

    def test_round_trip(self, store):
        tool = Neo4jMemoryRetrieverTool(memory_store=store, recall=RecallConfig(history_limit=3), name="recall")
        restored = Neo4jMemoryRetrieverTool.from_dict(tool.to_dict())
        assert (restored.name, restored.recall.history_limit) == ("recall", 3)
        assert restored.memory_store is Neo4jMemoryStore.from_dict(store.to_dict())


# ---------------------------------------------------------------- store_memory


class TestWriterTool:
    def test_stores_preference_fact_and_entity(self, store):
        client = attach(store, fake_client())
        client.long_term.add_preference.return_value = SimpleNamespace(id="p1")
        client.long_term.add_fact.return_value = SimpleNamespace(id="f1")
        entity = SimpleNamespace(id="e1", name="Northwind Capital", type="ORGANIZATION")
        client.long_term.add_entity = AsyncMock(return_value=(entity, SimpleNamespace(action="none")))
        tool = Neo4jMemoryWriterTool(memory_store=store)

        assert tool.store("preference", category="region", preference="Germany", user_id="u1") == (
            "Stored preference [region] Germany."
        )
        assert client.long_term.add_preference.await_args.kwargs["user_identifier"] == "u1"
        assert tool.store("fact", subject="Pravesh", predicate="works_at", object="Northwind") == (
            "Stored fact: Pravesh works_at Northwind."
        )
        assert tool.store("entity", name="Northwind Capital", entity_type="ORGANIZATION") == (
            "Stored entity Northwind Capital (ORGANIZATION)."
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"memory_type": "note"}, "memory_type must be one of"),
            ({"memory_type": "preference", "category": "region"}, "needs preference"),
            ({"memory_type": "fact", "subject": "a"}, "needs predicate, object"),
        ],
    )
    def test_bad_calls_are_explained_to_the_model(self, store, kwargs, message):
        store._connected = AsyncMock(side_effect=AssertionError("must not connect"))
        assert message in Neo4jMemoryWriterTool(memory_store=store).store(**kwargs)

    def test_nams_skips_preferences_and_facts_with_a_message(self, monkeypatch):
        monkeypatch.setenv("NAMS_API_KEY", "k")
        nams_store = Neo4jMemoryStore({"backend": "nams"})
        tool = Neo4jMemoryWriterTool(memory_store=nams_store)
        assert tool.store("preference", category="c", preference="p") == (
            "Not stored: this memory backend (NAMS) does not store preferences."
        )
        assert "does not store facts" in tool.store("fact", subject="a", predicate="b", object="c")
        nams_store._runner.stop()

    def test_round_trip(self, store):
        restored = Neo4jMemoryWriterTool.from_dict(Neo4jMemoryWriterTool(memory_store=store, name="remember").to_dict())
        assert restored.name == "remember"


# ---------------------------------------------------------------- a real Agent, with a scripted model


@component
class ScriptedChatGenerator:
    """Stands in for an LLM: replies with the next scripted message, recording what it was given."""

    def __init__(self, script: list[ChatMessage]) -> None:
        self.script = list(script)
        self.seen: list[list[ChatMessage]] = []

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        self.seen.append(list(messages))
        return {"replies": [self.script.pop(0)]}


def test_agent_uses_both_tools_with_state_injected(store):
    client = attach(store, fake_client())
    client.long_term.add_preference.return_value = SimpleNamespace(id="p1")
    client.long_term.get_preferences_for = AsyncMock(return_value=[preference("Focus on Germany")])

    generator = ScriptedChatGenerator(
        [
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall(
                        "store_memory",
                        {"memory_type": "preference", "category": "region", "preference": "Focus on Germany"},
                        id="c1",
                    )
                ]
            ),
            ChatMessage.from_assistant(tool_calls=[ToolCall("retrieve_memories", {}, id="c2")]),
            ChatMessage.from_assistant("You focus on Germany."),
        ]
    )
    agent = Agent(
        chat_generator=generator,
        tools=[Neo4jMemoryRetrieverTool(memory_store=store), Neo4jMemoryWriterTool(memory_store=store)],
        state_schema={"session_id": {"type": str}, "user_id": {"type": str}},
    )
    result = agent.run(messages=[ChatMessage.from_user("I focus on Germany.")], session_id="s1", user_id="pravesh")

    assert result["last_message"].text == "You focus on Germany."
    # user_id came from Agent State, not from the model's arguments
    assert client.long_term.add_preference.await_args.kwargs["user_identifier"] == "pravesh"
    client.long_term.get_preferences_for.assert_awaited_once_with("pravesh")
    tool_results = [r.result for m in result["messages"] for r in m.tool_call_results]
    assert tool_results[0] == "Stored preference [region] Focus on Germany."
    assert "User preferences:\n- [region] Focus on Germany" in tool_results[1]


def test_agent_with_memory_tools_saves_and_loads_as_yaml(store):
    agent = Agent(
        chat_generator=ScriptedChatGenerator([]),
        tools=[Neo4jMemoryRetrieverTool(memory_store=store), Neo4jMemoryWriterTool(memory_store=store)],
        state_schema={"session_id": {"type": str}, "user_id": {"type": str}},
    )
    pipe = Pipeline()
    pipe.add_component("agent", agent)
    yaml_text = pipe.dumps()
    assert "super-secret-password" not in yaml_text
    loaded = Pipeline.loads(yaml_text, allowed_modules=["tests.test_memory_tools"])  # only for the scripted model
    tools = loaded.get_component("agent").tools
    assert [t.name for t in tools] == ["retrieve_memories", "store_memory"]
    assert tools[0].memory_store is tools[1].memory_store  # one shared connection after loading
