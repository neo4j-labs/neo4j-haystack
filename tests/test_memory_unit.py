"""Unit tests for the agent memory store and components. No network: the MemoryClient is mocked."""

import asyncio
import json
import subprocess
import sys
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("neo4j_agent_memory")

from haystack import Pipeline
from haystack.dataclasses import ChatMessage, ToolCall
from neo4j_agent_memory.extraction.base import ExtractedEntity, ExtractedPreference, ExtractedRelation, ExtractionResult
from pydantic import ValidationError

from haystack_integrations.components.retrievers.neo4j import Neo4jMemoryRetriever
from haystack_integrations.components.writers.neo4j import Neo4jMemoryWriter
from haystack_integrations.memory_stores.neo4j import (
    ExtractionPolicy,
    Neo4jMemoryStore,
    RecallConfig,
    _convert,
    _long_term,
)
from haystack_integrations.memory_stores.neo4j._embedding import _SharedEmbedder
from haystack_integrations.memory_stores.neo4j._loop import _LoopRunner


@pytest.fixture(autouse=True)
def env(monkeypatch):
    # The test reviewer functions live in this module; a saved custom reviewer must be on Haystack's allowlist.
    monkeypatch.setenv("HAYSTACK_DESERIALIZATION_ALLOWLIST", __name__)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "super-secret-password")


def fake_client(*, extractor=None, linked=None):
    """A MemoryClient stand-in. ``linked`` is what the MENTIONS lookup returns."""
    message = SimpleNamespace(id=uuid.uuid4())
    client = SimpleNamespace(
        short_term=SimpleNamespace(add_message=AsyncMock(return_value=message), extractor=extractor),
        long_term=SimpleNamespace(
            search_preferences=AsyncMock(return_value=[]),
            search_entities=AsyncMock(return_value=[]),
            search_facts=AsyncMock(return_value=[]),
            add_preference=AsyncMock(),
            add_fact=AsyncMock(),
            get_entity_by_name=AsyncMock(side_effect=lambda name: SimpleNamespace(name=name)),
            add_relationship=AsyncMock(),
        ),
        reasoning=SimpleNamespace(
            get_similar_traces=AsyncMock(return_value=[]),
            get_trace_with_steps=AsyncMock(),
            start_trace=AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())),
            add_step=AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())),
            record_tool_call=AsyncMock(),
            complete_trace=AsyncMock(),
        ),
        query=SimpleNamespace(cypher=AsyncMock(return_value=linked or [])),
    )
    return client


@pytest.fixture
def store():
    s = Neo4jMemoryStore()
    yield s
    s._runner.stop()


def attach(store, client):
    store._connected = AsyncMock(return_value=client)
    return client


# ---------------------------------------------------------------- _LoopRunner


class TestLoopRunner:
    async def _thread_name(self):
        return threading.current_thread().name

    def test_sync_call_runs_on_private_thread(self):
        runner = _LoopRunner()
        assert runner.run(self._thread_name()) == "neo4j-agent-memory"
        runner.stop()

    def test_sync_call_inside_running_loop(self):
        runner = _LoopRunner()

        async def jupyter_cell():
            return runner.run(self._thread_name())  # sync call while this loop is running

        assert asyncio.run(jupyter_cell()) == "neo4j-agent-memory"
        runner.stop()

    def test_arun_from_another_loop(self):
        runner = _LoopRunner()
        assert asyncio.run(runner.arun(self._thread_name())) == "neo4j-agent-memory"
        runner.stop()

    def test_concurrent_threads_share_one_loop(self):
        runner = _LoopRunner()

        async def loop_id():
            return id(asyncio.get_running_loop())

        results = []
        threads = [threading.Thread(target=lambda: results.append(runner.run(loop_id()))) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(results)) == 1
        runner.stop()

    def test_stop_then_reuse(self):
        runner = _LoopRunner()

        async def current_loop():
            return asyncio.get_running_loop()

        first = runner.run(current_loop())
        runner.stop()
        assert first.is_closed()
        second = runner.run(current_loop())
        assert second is not first and second.is_running()
        runner.stop()

    def test_deadlock_guard(self):
        runner = _LoopRunner()

        async def nested():
            return runner.run(self._thread_name())

        with pytest.raises(RuntimeError, match="deadlock"):
            runner.run(nested())
        runner.stop()


# ---------------------------------------------------------------- config (ExtractionPolicy, RecallConfig)


def keep_orgs(result):
    return result.model_copy(update={"entities": [e for e in result.entities if e.type == "ORGANIZATION"]})


class TestPolicies:
    @pytest.mark.parametrize(
        "kwargs",
        [{"user": "explicit-ish"}, {"assistant": "tool"}, {"min_confidence": 1.5}, {"reviewer": "not callable"}],
    )
    def test_invalid_policy(self, kwargs):
        with pytest.raises(ValueError):
            ExtractionPolicy(**kwargs)

    def test_policy_round_trip_with_reviewer(self):
        policy = ExtractionPolicy(user="review", allowed_types=["ORGANIZATION"], min_confidence=0.5, reviewer=keep_orgs)
        data = policy.to_dict()
        assert data["reviewer"] == f"{__name__}.keep_orgs"
        json.dumps(data)
        restored = ExtractionPolicy.from_dict(data)
        assert restored.reviewer is keep_orgs
        assert restored == policy

    @pytest.mark.parametrize("kwargs", [{"history_limit": -1}, {"entity_threshold": 2.0}, {"trace_limit": True}])
    def test_invalid_recall(self, kwargs):
        with pytest.raises(ValueError):
            RecallConfig(**kwargs)

    def test_recall_round_trip(self):
        recall = RecallConfig(history_limit=3, template="{{ preferences }}")
        assert RecallConfig.from_dict(recall.to_dict()) == recall


# ---------------------------------------------------------------- _convert


class TestConvert:
    def test_role_mapping(self):
        def msg(role):
            return SimpleNamespace(id="m1", role=role, content="hi")

        assert _convert.to_chat_message(msg("user")).role.value == "user"
        assert _convert.to_chat_message(msg(SimpleNamespace(value="assistant"))).role.value == "assistant"
        assert _convert.to_chat_message(msg("system")) is None
        assert _convert.to_chat_message(msg("tool")) is None

    def test_entities_from_nested_tool_json(self):
        call = ToolCall(tool_name="t", arguments={})
        messages = [
            ChatMessage.from_tool(
                json.dumps(
                    {"hits": [{"name": "Acme", "type": "organization", "meta": {"name": "Bob", "type": "PERSON"}}]}
                ),
                origin=call,
            ),
            ChatMessage.from_tool(json.dumps([{"name": "acme", "type": "ORGANIZATION"}]), origin=call),  # duplicate
            ChatMessage.from_tool("not json", origin=call),
            ChatMessage.from_tool(json.dumps({"name": "Err", "type": "X"}), origin=call, error=True),  # failed tool
            ChatMessage.from_tool(json.dumps({"name": "NoType"}), origin=call),
        ]
        assert _convert.entities_from_tool_results(messages) == [
            {"name": "Acme", "type": "ORGANIZATION"},
            {"name": "Bob", "type": "PERSON"},
        ]

    def test_split_turn_drops_injected_history(self):
        transcript = [
            ChatMessage.from_system("s"),
            ChatMessage.from_user("old"),
            ChatMessage.from_assistant("old answer"),
            ChatMessage.from_user("now"),
            ChatMessage.from_assistant("final"),
        ]
        turn, answer = _convert.split_turn(transcript, None)
        assert [m.text for m in turn] == ["final"]
        assert answer.text == "final"

    def test_split_turn_with_replies_only(self):
        turn, answer = _convert.split_turn(None, [ChatMessage.from_assistant("a")])
        assert answer.text == "a" and len(turn) == 1

    def test_split_turn_without_answer(self):
        assert _convert.split_turn(None, [ChatMessage.from_assistant(tool_calls=[ToolCall("t", {})])])[1] is None

    def test_one_line(self):
        assert _convert.one_line("a\n  b\tc") == "a b c"
        assert len(_convert.one_line("x" * 300)) == 150


# ---------------------------------------------------------------- extraction policy -> library calls


def run(coro):
    return asyncio.run(coro)


class TestWriteWithPolicy:
    def write(self, client, mode, **kwargs):
        return run(
            _long_term.write_with_policy(
                client,
                session_id="s",
                role="user",
                text="Northwind and Contoso",
                mode=mode,
                policy=kwargs.pop("policy", ExtractionPolicy()),
                **kwargs,
            )
        )

    def test_auto(self):
        client = fake_client(linked=[{"name": "Northwind", "type": "ORGANIZATION"}])
        _, linked = self.write(client, "auto", user_id="u1")
        client.short_term.add_message.assert_awaited_once_with(
            "s", "user", "Northwind and Contoso", extraction_mode="auto", user_identifier="u1"
        )
        assert linked == [{"name": "Northwind", "type": "ORGANIZATION", "source": "extractor", "role": "user"}]

    def test_skip(self):
        client = fake_client()
        _, linked = self.write(client, "skip")
        assert client.short_term.add_message.await_args.kwargs["extraction_mode"] == "skip"
        assert linked == []
        client.query.cypher.assert_not_awaited()

    def test_explicit(self):
        client = fake_client(linked=[{"name": "Fabrikam", "type": "ORGANIZATION"}])
        self.write(client, "explicit", mentions=[{"name": "Fabrikam", "type": "organization"}, {"id": "e-1"}])
        kwargs = client.short_term.add_message.await_args.kwargs
        assert kwargs["extraction_mode"] == "explicit"
        refs = kwargs["explicit_mentions"]
        assert (refs[0].name, refs[0].type) == ("Fabrikam", "ORGANIZATION")
        assert refs[1].id == "e-1"

    def test_tool_results_with_and_without_entities(self):
        call = ToolCall(tool_name="t", arguments={})
        turn = [ChatMessage.from_tool(json.dumps({"name": "Acme", "type": "ORGANIZATION"}), origin=call)]
        client = fake_client(linked=[{"name": "Acme", "type": "ORGANIZATION"}])
        _, linked = self.write(client, "tool_results", turn=turn)
        assert client.short_term.add_message.await_args.kwargs["extraction_mode"] == "explicit"
        assert linked[0]["source"] == "tool_results"

        client = fake_client()
        _, linked = self.write(client, "tool_results", turn=[])
        assert client.short_term.add_message.await_args.kwargs["extraction_mode"] == "skip"
        assert linked == []

    def proposal(self):
        return ExtractionResult(
            entities=[
                ExtractedEntity(name="Northwind", type="ORGANIZATION", confidence=0.9),
                ExtractedEntity(name="Contoso", type="ORGANIZATION", confidence=0.9),
                ExtractedEntity(name="Pravesh", type="PERSON", confidence=0.2),
                ExtractedEntity(name="advertising", type="OBJECT", confidence=0.9),
            ],
            relations=[
                ExtractedRelation(source="Northwind", target="Contoso", relation_type="COMPETES_WITH", confidence=0.8),
                ExtractedRelation(source="Northwind", target="Pravesh", relation_type="EMPLOYS", confidence=0.9),
            ],
            preferences=[ExtractedPreference(category="region", preference="Germany", confidence=0.9)],
        )

    def test_review_filters_types_confidence_and_dangling_relations(self):
        extractor = SimpleNamespace(extract=AsyncMock(return_value=self.proposal()))
        client = fake_client(extractor=extractor, linked=[{"name": "Contoso", "type": "ORGANIZATION"}])
        policy = ExtractionPolicy(user="review", allowed_types=["organization", "person"], min_confidence=0.5)
        self.write(client, "review", policy=policy)

        refs = client.short_term.add_message.await_args.kwargs["explicit_mentions"]
        assert sorted(r.name for r in refs) == ["Contoso", "Northwind"]  # PERSON below confidence, OBJECT not allowed
        client.long_term.add_relationship.assert_awaited_once()
        assert client.long_term.add_relationship.await_args.args[2] == "COMPETES_WITH"
        assert client.long_term.add_relationship.await_args.kwargs["extractor"] == "reviewed"
        client.long_term.add_preference.assert_awaited_once()

    def test_review_reviewer_runs_after_filters_and_switches(self):
        seen = {}

        def reviewer(result):
            seen["names"] = [e.name for e in result.entities]
            return result.model_copy(update={"entities": result.entities[:1]})

        extractor = SimpleNamespace(extract=AsyncMock(return_value=self.proposal()))
        client = fake_client(extractor=extractor)
        policy = ExtractionPolicy(
            user="review", min_confidence=0.5, reviewer=reviewer, save_relations=False, save_preferences=False
        )
        self.write(client, "review", policy=policy)
        assert seen["names"] == ["Northwind", "Contoso", "advertising"]
        assert [r.name for r in client.short_term.add_message.await_args.kwargs["explicit_mentions"]] == ["Northwind"]
        client.long_term.add_relationship.assert_not_awaited()
        client.long_term.add_preference.assert_not_awaited()

    def test_review_extraction_error_saves_with_skip(self):
        extractor = SimpleNamespace(extract=AsyncMock(side_effect=RuntimeError("LLM down")))
        client = fake_client(extractor=extractor)
        _, linked = self.write(client, "review", policy=ExtractionPolicy(user="review"))
        assert client.short_term.add_message.await_args.kwargs["extraction_mode"] == "skip"
        assert linked == []
        with pytest.raises(RuntimeError):
            self.write(fake_client(extractor=extractor), "review", policy=ExtractionPolicy(user="review"), strict=True)

    def test_review_without_extractor_saves_with_skip(self):
        client = fake_client(extractor=None)
        self.write(client, "review", policy=ExtractionPolicy(user="review"))
        assert client.short_term.add_message.await_args.kwargs["extraction_mode"] == "skip"


# ---------------------------------------------------------------- retriever


def pref(text, sim=0.8):
    return SimpleNamespace(
        id=uuid.uuid4(), category="region", preference=text, context=None, confidence=1.0, metadata={"similarity": sim}
    )


class TestRetriever:
    def test_messages_order_and_outputs(self, store):
        client = attach(store, fake_client())
        client.query.cypher.return_value = [
            {"id": "1", "role": "user", "content": "earlier"},
            {"id": "2", "role": "assistant", "content": "reply"},
        ]
        client.long_term.search_preferences.return_value = [pref("Focus on Germany")]
        trace = SimpleNamespace(
            id=uuid.uuid4(), task="find companies", outcome="Acme", success=True, metadata={"similarity": 0.9}
        )
        client.reasoning.get_similar_traces.return_value = [trace]
        step = SimpleNamespace(tool_calls=[SimpleNamespace(tool_name="search"), SimpleNamespace(tool_name="search")])
        client.reasoning.get_trace_with_steps.return_value = SimpleNamespace(steps=[step])

        out = Neo4jMemoryRetriever(store, system_prompt="sys").run(query="now", session_id="s")
        assert out["messages"][0].text == "sys"
        assert out["messages"][1].meta == {"neo4j_memory": "recall"}
        assert [m.text for m in out["messages"]][2:] == ["earlier", "reply", "now"]
        assert "- [region] Focus on Germany" in out["context"]
        assert "- find companies -> Acme (tools: search)" in out["context"]
        assert out["memories"]["traces"][0]["tools"] == ["search"]
        assert set(out["memories"]) == {"preferences", "entities", "facts", "traces"}
        json.dumps(out["memories"])

    def test_no_memory_message_when_nothing_recalled(self, store):
        attach(store, fake_client())
        out = Neo4jMemoryRetriever(store).run(query="q", session_id="s")
        assert out["context"] == ""
        assert [m.text for m in out["messages"]] == ["q"]

    def test_read_switches(self, store):
        client = attach(store, fake_client())
        recall = RecallConfig(read_short_term=False, read_long_term=False, read_reasoning=False)
        Neo4jMemoryRetriever(store, recall=recall).run(query="q", session_id="s")
        client.query.cypher.assert_not_awaited()
        client.long_term.search_preferences.assert_not_awaited()
        client.reasoning.get_similar_traces.assert_not_awaited()

    def test_failed_traces_are_recalled_with_reasons(self, store):
        client = attach(store, fake_client())
        ok = SimpleNamespace(
            id="t1", task="German AI", outcome="Aleph Alpha", success=True, metadata={"similarity": 0.9}
        )
        bad = SimpleNamespace(
            id="t2", task="French AI", outcome="No answer.", success=False, metadata={"similarity": 0.8}
        )
        client.reasoning.get_similar_traces.return_value = [ok, bad]
        steps = {
            "t1": [
                SimpleNamespace(
                    tool_calls=[SimpleNamespace(tool_name="search", status="success", error=None, result="[]")]
                )
            ],
            "t2": [
                SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            tool_name="search", status=SimpleNamespace(value="error"), error="HTTP 503", result=None
                        )
                    ]
                )
            ],
        }
        client.reasoning.get_trace_with_steps.side_effect = lambda trace_id: SimpleNamespace(steps=steps[trace_id])

        async def cypher(query, params):
            if "error_kind" in query:
                return [{"id": "t2", "error_kind": "tool_error"}]
            return []  # history

        client.query.cypher = AsyncMock(side_effect=cypher)
        out = Neo4jMemoryRetriever(store).run(query="AI companies", session_id="s")

        assert client.reasoning.get_similar_traces.await_args.kwargs["success_only"] is False
        failed = out["memories"]["traces"][1]
        assert (failed["success"], failed["error_kind"], failed["errors"]) == (
            False,
            "tool_error",
            [{"tool": "search", "error": "HTTP 503"}],
        )
        assert out["memories"]["traces"][0]["error_kind"] is None
        assert "Similar past tasks that succeeded:\n- German AI -> Aleph Alpha (tools: search)" in out["context"]
        assert "- French AI -> No answer. [tool_error] search: HTTP 503" in out["context"]

    def test_successful_traces_only(self, store):
        client = attach(store, fake_client())
        Neo4jMemoryRetriever(store, recall=RecallConfig(include_failed_traces=False)).run(query="q", session_id="s")
        assert client.reasoning.get_similar_traces.await_args.kwargs["success_only"] is True

    def test_thresholds_and_trace_tools_switch(self, store):
        client = attach(store, fake_client())
        client.reasoning.get_similar_traces.return_value = [
            SimpleNamespace(id=1, task="t", outcome="o", success=True, metadata={})
        ]
        recall = RecallConfig(preference_threshold=0.11, include_facts=False, include_trace_tools=False)
        Neo4jMemoryRetriever(store, recall=recall).run(query="q", session_id="s")
        assert client.long_term.search_preferences.await_args.kwargs["threshold"] == 0.11
        client.long_term.search_facts.assert_not_awaited()
        client.reasoning.get_trace_with_steps.assert_not_awaited()

    def test_template_override(self, store):
        client = attach(store, fake_client())
        client.long_term.search_preferences.return_value = [pref("A"), pref("B")]
        recall = RecallConfig(template="Prefs: {{ preferences | map(attribute='preference') | join('; ') }}")
        assert Neo4jMemoryRetriever(store, recall=recall).run(query="q", session_id="s")["context"] == "Prefs: A; B"

    def test_run_async(self, store):
        attach(store, fake_client())
        out = asyncio.run(Neo4jMemoryRetriever(store).run_async(query="q", session_id="s"))
        assert [m.text for m in out["messages"]] == ["q"]

    def test_failure_degrades_or_raises(self, store):
        store._connected = AsyncMock(side_effect=ConnectionError("down"))
        out = Neo4jMemoryRetriever(store, system_prompt="sys").run(query="q", session_id="s")
        assert [m.text for m in out["messages"]] == ["sys", "q"]
        store.strict = True
        with pytest.raises(ConnectionError):
            Neo4jMemoryRetriever(store).run(query="q", session_id="s")

    def test_rejects_non_store(self):
        with pytest.raises(ValueError, match="Neo4jMemoryStore"):
            Neo4jMemoryRetriever(object())


# ---------------------------------------------------------------- writer


class TestWriter:
    def test_replies_turn_writes_query_and_answer(self, store):
        client = attach(store, fake_client())
        writer = Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="skip", assistant="skip"))
        out = writer.run(query="q", session_id="s", replies=[ChatMessage.from_assistant("a")], user_id="u")
        calls = [(c.args[1], c.args[2]) for c in client.short_term.add_message.await_args_list]
        assert calls == [("user", "q"), ("assistant", "a")]
        assert all(c.kwargs["user_identifier"] == "u" for c in client.short_term.add_message.await_args_list)
        assert out["messages_written"] == 2
        assert out["trace_id"] is not None

    def test_explicit_preferences_and_facts(self, store):
        client = attach(store, fake_client())
        Neo4jMemoryWriter(store, write_short_term=False, record_reasoning="never").run(
            query="q",
            session_id="s",
            preferences=[{"category": "region", "preference": "Germany", "context": "investing"}],
            facts=[{"subject": "Pravesh", "predicate": "works_at", "object": "Northwind"}],
        )
        client.short_term.add_message.assert_not_awaited()
        assert client.long_term.add_preference.await_args.args == ("region", "Germany")
        assert client.long_term.add_preference.await_args.kwargs["context"] == "investing"
        assert client.long_term.add_fact.await_args.args == ("Pravesh", "works_at", "Northwind")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"preferences": [{"category": "x"}]},
            {"facts": [{"subject": "a", "predicate": "b"}]},
            {"mentions": [{"type": "PERSON"}]},
        ],
    )
    def test_invalid_items_always_raise(self, store, kwargs):
        attach(store, fake_client())
        with pytest.raises(ValueError):
            Neo4jMemoryWriter(store).run(query="q", session_id="s", **kwargs)

    def test_reasoning_steps_and_success_rule(self, store):
        client = attach(store, fake_client())
        ok, bad = ToolCall("search", {"q": 1}, id="c1"), ToolCall("lookup", {}, id="c2")
        transcript = [
            ChatMessage.from_user("q"),
            ChatMessage.from_assistant("thinking", tool_calls=[ok, bad]),
            ChatMessage.from_tool("[]", origin=ok),
            ChatMessage.from_tool("boom", origin=bad, error=True),
            ChatMessage.from_assistant("done"),
        ]
        Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="skip", assistant="skip")).run(
            query="q", session_id="s", messages=transcript
        )
        rs = client.reasoning
        assert rs.add_step.await_args.kwargs == {"thought": "thinking", "action": "search, lookup"}
        statuses = [
            (c.args[1], c.kwargs["status"].value, c.kwargs["error"]) for c in rs.record_tool_call.await_args_list
        ]
        assert statuses == [("search", "success", None), ("lookup", "error", "boom")]
        outcome = rs.complete_trace.await_args.kwargs["outcome"]
        assert (outcome.success, outcome.summary, outcome.error_kind) == (False, "done", "tool_error")
        assert outcome.metrics == {"tool_calls": 2.0, "tool_errors": 1.0}

    def test_success_input_wins(self, store):
        client = attach(store, fake_client())
        Neo4jMemoryWriter(store).run(
            query="q", session_id="s", replies=[ChatMessage.from_assistant("a")], success=False
        )
        outcome = client.reasoning.complete_trace.await_args.kwargs["outcome"]
        assert (outcome.success, outcome.error_kind) == (False, "marked_failed")

    def test_no_answer_is_recorded_as_failed(self, store):
        client = attach(store, fake_client())
        out = Neo4jMemoryWriter(store).run(query="q", session_id="s", replies=[])
        outcome = client.reasoning.complete_trace.await_args.kwargs["outcome"]
        assert out["trace_id"] is not None
        assert (outcome.success, outcome.summary, outcome.error_kind) == (False, "No answer.", "no_answer")

    def test_tool_call_without_result_is_a_tool_error(self, store):
        client = attach(store, fake_client())
        call = ToolCall("lookup", {}, id="c1")
        transcript = [ChatMessage.from_user("q"), ChatMessage.from_assistant(tool_calls=[call])]
        Neo4jMemoryWriter(store).run(query="q", session_id="s", messages=transcript)
        recorded = client.reasoning.record_tool_call.await_args.kwargs
        assert (recorded["status"].value, recorded["error"]) == ("error", "No result returned.")
        outcome = client.reasoning.complete_trace.await_args.kwargs["outcome"]
        assert (outcome.success, outcome.error_kind) == (False, "tool_error")

    def test_recovered_tool_error_with_success_true_has_no_error_kind(self, store):
        client = attach(store, fake_client())
        call = ToolCall("lookup", {}, id="c1")
        transcript = [
            ChatMessage.from_user("q"),
            ChatMessage.from_assistant(tool_calls=[call]),
            ChatMessage.from_tool("timeout", origin=call, error=True),
            ChatMessage.from_assistant("Answered from cache."),
        ]
        Neo4jMemoryWriter(store).run(query="q", session_id="s", messages=transcript, success=True)
        outcome = client.reasoning.complete_trace.await_args.kwargs["outcome"]
        assert (outcome.success, outcome.error_kind, outcome.metrics["tool_errors"]) == (True, None, 1.0)

    @pytest.mark.parametrize(
        ("mode", "tools", "recorded"),
        [("always", False, True), ("with_tools", False, False), ("with_tools", True, True), ("never", True, False)],
    )
    def test_record_reasoning_modes(self, store, mode, tools, recorded):
        client = attach(store, fake_client())
        call = ToolCall("t", {}, id="c")
        messages = [ChatMessage.from_user("q")]
        if tools:
            messages += [ChatMessage.from_assistant(tool_calls=[call]), ChatMessage.from_tool("{}", origin=call)]
        messages.append(ChatMessage.from_assistant("a"))
        out = Neo4jMemoryWriter(store, record_reasoning=mode).run(query="q", session_id="s", messages=messages)
        assert (out["trace_id"] is not None) is recorded
        assert client.reasoning.start_trace.await_count == int(recorded)

    def test_failure_degrades_with_progress_or_raises(self, store):
        client = attach(store, fake_client())
        client.long_term.add_fact = AsyncMock(side_effect=RuntimeError("write failed"))
        writer = Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="skip", assistant="skip"))
        out = writer.run(query="q", session_id="s", facts=[{"subject": "a", "predicate": "b", "object": "c"}])
        assert out == {"messages_written": 1, "trace_id": None, "entities": []}
        store.strict = True
        with pytest.raises(RuntimeError):
            writer.run(query="q", session_id="s", facts=[{"subject": "a", "predicate": "b", "object": "c"}])

    def test_run_async(self, store):
        attach(store, fake_client())
        out = asyncio.run(Neo4jMemoryWriter(store, record_reasoning="never").run_async(query="q", session_id="s"))
        assert out["messages_written"] == 1

    def test_invalid_record_reasoning(self, store):
        with pytest.raises(ValueError):
            Neo4jMemoryWriter(store, record_reasoning="sometimes")


# ---------------------------------------------------------------- store and serialization


class TestSerialization:
    def test_store_validates_library_config(self):
        with pytest.raises(ValidationError):
            Neo4jMemoryStore(extraction={"extractor_type": "magic"})

    def test_store_settings_are_explicit(self, store, monkeypatch):
        monkeypatch.setenv("NAM_NEO4J__URI", "bolt://elsewhere:7687")
        settings = store._settings()
        assert settings.neo4j.uri == "bolt://localhost:7687"
        assert settings.neo4j.password.get_secret_value() == "super-secret-password"
        assert settings.backend == "bolt"
        assert settings.extraction.extractor_type.value == "llm"

    def test_store_to_dict_has_no_secret_values(self, store):
        data = store.to_dict()
        assert data["init_parameters"]["password"] == {
            "type": "env_var",
            "env_vars": ["NEO4J_PASSWORD"],
            "strict": True,
        }
        assert "super-secret-password" not in json.dumps(data)

    def test_store_round_trip_reuses_instance(self):
        data = Neo4jMemoryStore(database="mem", memory={"multi_tenant": True}).to_dict()
        first = Neo4jMemoryStore.from_dict(data)
        assert first.database == "mem" and first.memory == {"multi_tenant": True}
        assert Neo4jMemoryStore.from_dict(data) is first
        assert data["init_parameters"]["password"]["type"] == "env_var"  # input not mutated

    def test_components_round_trip(self, store):
        retriever = Neo4jMemoryRetriever(store, recall=RecallConfig(history_limit=4), system_prompt="sys")
        writer = Neo4jMemoryWriter(
            store, extraction_policy=ExtractionPolicy(user="review", reviewer=keep_orgs), record_reasoning="with_tools"
        )
        r2 = Neo4jMemoryRetriever.from_dict(retriever.to_dict())
        w2 = Neo4jMemoryWriter.from_dict(writer.to_dict())
        assert r2.recall.history_limit == 4 and r2.system_prompt == "sys"
        assert w2.extraction_policy.reviewer is keep_orgs and w2.record_reasoning == "with_tools"
        assert r2.store is w2.store

    def test_pipeline_dumps_loads_shares_one_store(self, store):
        pipe = Pipeline()
        pipe.add_component("reader", Neo4jMemoryRetriever(store))
        pipe.add_component("writer", Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="review")))
        yaml_text = pipe.dumps()
        assert "super-secret-password" not in yaml_text
        loaded = Pipeline.loads(yaml_text)  # default allowlist
        assert loaded.get_component("reader").store is loaded.get_component("writer").store
        assert loaded.get_component("writer").extraction_policy.user == "review"


def test_missing_memory_extra_gives_install_hint():
    script = (
        "import sys; sys.modules['neo4j_agent_memory'] = None\n"
        "from haystack_integrations.components.retrievers.neo4j import Neo4jMemoryRetriever\n"
        "from haystack_integrations.memory_stores.neo4j import Neo4jMemoryStore\n"
        "try:\n"
        "    Neo4jMemoryStore()\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert 'pip install "haystack-neo4j[memory]"' in proc.stdout


# ---------------------------------------------------------------- shared embedder


class CountingEmbedder:
    dimensions = 3
    model = "counting"

    def __init__(self, fail_times=0):
        self.calls, self.batches, self.fail_times = [], [], fail_times

    async def embed(self, text):
        self.calls.append(text)
        await asyncio.sleep(0.01)
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("rate limited")
        return [float(len(text)), 0.0, 1.0]

    async def embed_batch(self, texts):
        self.batches.append(list(texts))
        return [[float(len(t)), 0.0, 1.0] for t in texts]


class TestSharedEmbedder:
    def test_concurrent_identical_texts_embed_once(self):
        inner = CountingEmbedder()
        shared = _SharedEmbedder(inner)

        async def four_searches():
            return await asyncio.gather(*(shared.embed("same query") for _ in range(4)))

        vectors = asyncio.run(four_searches())
        assert inner.calls == ["same query"]
        assert all(v == vectors[0] for v in vectors)
        vectors[0].append(99.0)  # callers get their own copy
        assert asyncio.run(shared.embed("same query")) == [10.0, 0.0, 1.0]

    def test_attributes_pass_through(self):
        shared = _SharedEmbedder(CountingEmbedder())
        assert (shared.dimensions, shared.model) == (3, "counting")

    def test_failures_are_not_cached(self):
        inner = CountingEmbedder(fail_times=1)
        shared = _SharedEmbedder(inner)
        with pytest.raises(RuntimeError):
            asyncio.run(shared.embed("q"))
        assert asyncio.run(shared.embed("q")) == [1.0, 0.0, 1.0]
        assert inner.calls == ["q", "q"]

    def test_batch_embeds_only_missing_texts_once(self):
        inner = CountingEmbedder()
        shared = _SharedEmbedder(inner)

        async def go():
            await shared.embed("a")
            return await shared.embed_batch(["a", "bb", "bb", "ccc"])

        assert asyncio.run(go()) == [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]
        assert inner.batches == [["bb", "ccc"]]

    def test_lru_eviction(self):
        inner = CountingEmbedder()
        shared = _SharedEmbedder(inner, max_items=2)

        async def go():
            for text in ["a", "b", "a", "c", "a", "b"]:
                await shared.embed(text)

        asyncio.run(go())
        assert inner.calls == ["a", "b", "c", "b"]  # "b" was evicted by "c"; "a" stayed hot

    def test_store_wraps_the_default_embedder(self, store):
        client = store._create_client()
        assert isinstance(client._embedder_override, _SharedEmbedder)

    def test_library_accepts_the_wrapper_as_an_embedder(self):
        from neo4j_agent_memory.embeddings.base import adapt_to_legacy_embedder

        shared = _SharedEmbedder(CountingEmbedder())
        assert adapt_to_legacy_embedder(shared) is shared  # kept as-is, not rejected or re-wrapped
