"""
Integration tests for the agent memory components against a real Neo4j 5. No LLM or embedding API is used.

Start Neo4j with:
    docker run -d --name neo4j-mem -p 7687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5-community
Then run: pytest -m neo4j

Override the target with NEO4J_MEMORY_TEST_URI / NEO4J_MEMORY_TEST_PASSWORD. Tests use unique session ids and
never delete data they did not write, but point them at a throwaway database.
"""

import asyncio
import json
import os
import uuid

import pytest

pytest.importorskip("neo4j_agent_memory")

import neo4j
from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.components.retrievers.neo4j import Neo4jMemoryRetriever
from haystack_integrations.components.writers.neo4j import Neo4jMemoryWriter
from haystack_integrations.memory_stores.neo4j import ExtractionPolicy, Neo4jMemoryStore, RecallConfig
from haystack_integrations.memory_stores.neo4j import store as store_module
from haystack_integrations.memory_stores.neo4j._embedding import _SharedEmbedder

from .memory_fakes import FakeEmbedder, FakeExtractor

pytestmark = pytest.mark.neo4j

URI = os.environ.get("NEO4J_MEMORY_TEST_URI", "bolt://localhost:7687")
PASSWORD = os.environ.get("NEO4J_MEMORY_TEST_PASSWORD", "testpassword")
EMBEDDERS: list[FakeEmbedder] = []
OPEN_RECALL = RecallConfig(preference_threshold=0.95, entity_threshold=0.95, fact_threshold=0.95, trace_threshold=0.95)


def _neo4j_available() -> bool:
    try:
        with neo4j.GraphDatabase.driver(URI, auth=("neo4j", PASSWORD)) as driver:
            driver.verify_connectivity()
        return True
    except Exception:
        return False


if not _neo4j_available():
    pytest.skip(f"No Neo4j at {URI}; see the module docstring.", allow_module_level=True)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", URI)
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", PASSWORD)
    # Inject the fakes into every client the store creates, including stores built by Pipeline.loads.
    original = Neo4jMemoryStore._create_client

    def create_client(self):
        from neo4j_agent_memory import MemoryClient

        embedder = FakeEmbedder()
        EMBEDDERS.append(embedder)
        return MemoryClient(self._settings(), embedder=_SharedEmbedder(embedder), extractor=FakeExtractor())

    monkeypatch.setattr(Neo4jMemoryStore, "_create_client", create_client)
    yield
    monkeypatch.setattr(Neo4jMemoryStore, "_create_client", original)


@pytest.fixture
def store():
    s = Neo4jMemoryStore({"extraction": {"extractor_type": "none"}})
    yield s
    s.close()


@pytest.fixture
def cypher():
    driver = neo4j.GraphDatabase.driver(URI, auth=("neo4j", PASSWORD))

    def run(query, **params):
        records, _, _ = driver.execute_query(query, params)
        return [r.data() for r in records]

    yield run
    driver.close()


def sid() -> str:
    return f"test-{uuid.uuid4()}"


@component
class EchoGenerator:
    """Stands in for a chat generator: answers with the last user message, upper-cased."""

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]):
        last = next(m for m in reversed(messages) if m.is_from("user"))
        return {"replies": [ChatMessage.from_assistant(f"ANSWER: {last.text.upper()}")]}


# ---------------------------------------------------------------- short-term


def test_session_written_read_back_and_isolated(store):
    s1, s2 = sid(), sid()
    writer = Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="skip", assistant="skip"))
    retriever = Neo4jMemoryRetriever(store, recall=RecallConfig(read_long_term=False, read_reasoning=False))

    out = writer.run(query="Hello there", session_id=s1, replies=[ChatMessage.from_assistant("Hi!")])
    assert out["messages_written"] == 2

    first = retriever.run(query="And now?", session_id=s1)
    assert [(m.role.value, m.text) for m in first["history"]] == [("user", "Hello there"), ("assistant", "Hi!")]
    assert [m.text for m in first["messages"]][-1] == "And now?"
    assert first["history"][0].meta["neo4j_memory"] == "history"

    other = retriever.run(query="And now?", session_id=s2)
    assert other["history"] == []


def test_history_limit(store):
    s = sid()
    writer = Neo4jMemoryWriter(
        store, extraction_policy=ExtractionPolicy(user="skip", assistant="skip"), record_reasoning="never"
    )
    for i in range(3):
        writer.run(query=f"q{i}", session_id=s, replies=[ChatMessage.from_assistant(f"a{i}")])
    retriever = Neo4jMemoryRetriever(
        store, recall=RecallConfig(history_limit=2, read_long_term=False, read_reasoning=False)
    )
    assert [m.text for m in retriever.run(query="x", session_id=s)["history"]] == ["q2", "a2"]


# ---------------------------------------------------------------- long-term


def test_explicit_preferences_and_facts_are_recalled_in_a_new_session(store):
    tag = uuid.uuid4().hex[:8]
    preference = f"Focus on companies in Germany {tag}"
    fact_subject = f"Pravesh-{tag}"
    writer = Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="skip", assistant="skip"))
    writer.run(
        query="Remember this",
        session_id=sid(),
        preferences=[{"category": "region", "preference": preference}],
        facts=[{"subject": fact_subject, "predicate": "works_at", "object": "Northwind Capital"}],
    )

    retriever = Neo4jMemoryRetriever(store, recall=OPEN_RECALL)
    # The library embeds "category: preference"; the hash-based fake embedder needs that exact text.
    prefs = retriever.run(query=f"region: {preference}", session_id=sid())
    assert any(p["preference"] == preference for p in prefs["memories"]["preferences"])
    assert f"[region] {preference}" in prefs["context"]
    assert prefs["messages"][0].meta["neo4j_memory"] == "recall"

    facts = retriever.run(query=f"{fact_subject} works_at Northwind Capital", session_id=sid())
    assert any(f["subject"] == fact_subject for f in facts["memories"]["facts"])


def test_auto_mode_links_extractor_entities(store, cypher):
    s = sid()
    writer = Neo4jMemoryWriter(
        store, extraction_policy=ExtractionPolicy(user="auto", assistant="skip"), record_reasoning="never"
    )
    out = writer.run(query="Northwind competes with Contoso", session_id=s, replies=[ChatMessage.from_assistant("ok")])
    linked = {(e["name"], e["source"], e["role"]) for e in out["entities"]}
    assert {("Northwind", "extractor", "user"), ("Contoso", "extractor", "user")} <= linked
    rows = cypher(
        "MATCH (:Conversation {session_id: $s})-[:HAS_MESSAGE]->(m:Message {role: 'user'})-[:MENTIONS]->(e:Entity) "
        "RETURN collect(e.name) AS names",
        s=s,
    )
    assert {"Northwind", "Contoso"} <= set(rows[0]["names"])


def test_review_mode_filters_and_writes_relations_and_preferences(store, cypher):
    s = sid()
    text = "I prefer Northwind over Contoso for advertising, says Pravesh"
    policy = ExtractionPolicy(user="review", assistant="skip", min_confidence=0.5)
    out = Neo4jMemoryWriter(store, extraction_policy=policy, record_reasoning="never").run(query=text, session_id=s)

    # Pravesh (confidence 0.3) is dropped; entity types would be limited by the extractor's entity_types.
    assert sorted(e["name"] for e in out["entities"]) == ["Contoso", "Northwind", "advertising"]
    assert {e["source"] for e in out["entities"]} == {"review"}
    names = cypher(
        "MATCH (:Conversation {session_id: $s})-[:HAS_MESSAGE]->(m:Message)-[:MENTIONS]->(e:Entity) "
        "RETURN collect(e.name) AS n",
        s=s,
    )[0]["n"]
    assert sorted(names) == ["Contoso", "Northwind", "advertising"]
    rel = cypher(
        "MATCH (:Entity {name: 'Northwind'})-[r:RELATED_TO]->(:Entity {name: 'Contoso'}) "
        "WHERE r.extractor = 'reviewed' RETURN count(r) AS n"
    )
    assert rel[0]["n"] >= 1
    pref = cypher("MATCH (p:Preference {preference: $t}) RETURN count(p) AS n", t=text)
    assert pref[0]["n"] == 1


def reject_objects(result):
    """Module-level reviewer: drop OBJECT entities (descriptions the extractor promoted)."""
    return result.model_copy(update={"entities": [e for e in result.entities if e.type != "OBJECT"]})


def test_review_mode_custom_reviewer(store):
    policy = ExtractionPolicy(user="review", assistant="skip", reviewer=reject_objects, save_preferences=False)
    out = Neo4jMemoryWriter(store, extraction_policy=policy, record_reasoning="never").run(
        query="Northwind does advertising", session_id=sid()
    )
    assert [e["name"] for e in out["entities"]] == ["Northwind"]


def test_explicit_mode_links_only_given_mentions(store):
    s = sid()
    tag = uuid.uuid4().hex[:6]
    writer = Neo4jMemoryWriter(
        store, extraction_policy=ExtractionPolicy(user="explicit", assistant="skip"), record_reasoning="never"
    )
    out = writer.run(
        query="Northwind competes with Contoso",  # the extractor would find both; explicit mode ignores it
        session_id=s,
        mentions=[{"name": f"Fabrikam-{tag}", "type": "organization"}],
    )
    assert out["entities"] == [
        {"name": f"Fabrikam-{tag}", "type": "ORGANIZATION", "source": "explicit", "role": "user"}
    ]


def test_skip_mode_links_nothing(store, cypher):
    s = sid()
    writer = Neo4jMemoryWriter(
        store, extraction_policy=ExtractionPolicy(user="skip", assistant="skip"), record_reasoning="never"
    )
    out = writer.run(query="Northwind competes with Contoso", session_id=s)
    assert out["entities"] == []
    rows = cypher(
        "MATCH (:Conversation {session_id: $s})-[:HAS_MESSAGE]->(:Message)-[r:MENTIONS]->() RETURN count(r) AS n", s=s
    )
    assert rows[0]["n"] == 0


def test_assistant_tool_results_and_trace_graph(store, cypher):
    s = sid()
    tag = uuid.uuid4().hex[:6]
    query = f"Find German AI companies {tag}"
    call = ToolCall(tool_name="search_companies", arguments={"country": "DE"}, id="call-1")
    transcript = [
        ChatMessage.from_system("system prompt"),
        ChatMessage.from_user("an old question from history"),
        ChatMessage.from_assistant("an old answer"),
        ChatMessage.from_user(query),
        ChatMessage.from_assistant(tool_calls=[call]),
        ChatMessage.from_tool(json.dumps({"results": [{"name": f"Aleph-{tag}", "type": "Organization"}]}), origin=call),
        ChatMessage.from_assistant(f"Look at Aleph-{tag}."),
    ]
    writer = Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="skip", assistant="tool_results"))
    out = writer.run(query=query, session_id=s, messages=transcript)

    # Only the query and the final answer are stored; injected history is not saved twice.
    assert out["messages_written"] == 2
    stored = cypher(
        "MATCH (:Conversation {session_id: $s})-[:HAS_MESSAGE]->(m:Message) RETURN m.content AS c ORDER BY m.timestamp",
        s=s,
    )
    assert [r["c"] for r in stored] == [query, f"Look at Aleph-{tag}."]
    assert out["entities"] == [
        {"name": f"Aleph-{tag}", "type": "ORGANIZATION", "source": "tool_results", "role": "assistant"}
    ]

    trace = cypher(
        "MATCH (t:ReasoningTrace {id: $id})-[:HAS_STEP]->(st:ReasoningStep)"
        "-[:USES_TOOL]->(tc:ToolCall)-[:INSTANCE_OF]->(tool:Tool) "
        "RETURN t.task AS task, t.success AS success, st.action AS action, tc.status AS status, tool.name AS tool",
        id=out["trace_id"],
    )
    assert trace == [
        {"task": query, "success": True, "action": "search_companies", "status": "success", "tool": "search_companies"}
    ]

    recalled = Neo4jMemoryRetriever(store, recall=OPEN_RECALL).run(query=query, session_id=sid())
    traces = [t for t in recalled["memories"]["traces"] if t["id"] == out["trace_id"]]
    assert traces and traces[0]["tools"] == ["search_companies"]
    assert "(tools: search_companies)" in recalled["context"]


def test_failed_tool_marks_trace_unsuccessful(store, cypher):
    call = ToolCall(tool_name="lookup", arguments={}, id="c1")
    transcript = [
        ChatMessage.from_user("q"),
        ChatMessage.from_assistant(tool_calls=[call]),
        ChatMessage.from_tool("boom", origin=call, error=True),
        ChatMessage.from_assistant("Sorry, that failed."),
    ]
    out = Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="skip", assistant="skip")).run(
        query="q", session_id=sid(), messages=transcript
    )
    row = cypher(
        "MATCH (t:ReasoningTrace {id: $id})-[:HAS_STEP]->()-[:USES_TOOL]->(tc:ToolCall) "
        "RETURN t.success AS ok, t.error_kind AS kind, t.outcome AS outcome, tc.status AS status, tc.error AS error",
        id=out["trace_id"],
    )
    assert row == [
        {"ok": False, "kind": "tool_error", "outcome": "Sorry, that failed.", "status": "error", "error": "boom"}
    ]


def test_failed_and_unanswered_traces_are_recalled_with_reasons(store):
    tag = uuid.uuid4().hex[:8]
    writer = Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="skip", assistant="skip"))
    call = ToolCall(tool_name="search_companies", arguments={"country": "FR"}, id="c1")
    tool_failure = f"Find French AI companies {tag}"
    failed = writer.run(
        query=tool_failure,
        session_id=sid(),
        messages=[
            ChatMessage.from_user(tool_failure),
            ChatMessage.from_assistant(tool_calls=[call]),
            ChatMessage.from_tool("HTTP 503 from company API", origin=call, error=True),
        ],
    )
    no_answer = f"Summarise the report {tag}"
    unanswered = writer.run(query=no_answer, session_id=sid(), replies=[])

    retriever = Neo4jMemoryRetriever(store, recall=OPEN_RECALL)
    traces = {t["id"]: t for t in retriever.run(query=tool_failure, session_id=sid())["memories"]["traces"]}
    recalled = traces[failed["trace_id"]]
    assert (recalled["success"], recalled["error_kind"], recalled["outcome"]) == (False, "tool_error", "No answer.")
    assert recalled["errors"] == [{"tool": "search_companies", "error": "HTTP 503 from company API"}]

    context = retriever.run(query=no_answer, session_id=sid())
    assert any(
        t["id"] == unanswered["trace_id"] and t["error_kind"] == "no_answer" for t in context["memories"]["traces"]
    )
    assert "Similar past tasks that failed" in context["context"]

    only_ok = Neo4jMemoryRetriever(
        store, recall=RecallConfig(**{**OPEN_RECALL.to_dict(), "include_failed_traces": False})
    )
    assert all(t["success"] for t in only_ok.run(query=tool_failure, session_id=sid())["memories"]["traces"])


def test_record_reasoning_with_tools_skips_plain_turns(store):
    writer = Neo4jMemoryWriter(
        store, extraction_policy=ExtractionPolicy(user="skip", assistant="skip"), record_reasoning="with_tools"
    )
    out = writer.run(query="hi", session_id=sid(), replies=[ChatMessage.from_assistant("hello")])
    assert out["trace_id"] is None


# ---------------------------------------------------------------- pipelines and serialization


def _pipeline(store):
    pipe = Pipeline()
    pipe.add_component("reader", Neo4jMemoryRetriever(store, system_prompt="You are helpful.", recall=OPEN_RECALL))
    pipe.add_component("llm", EchoGenerator())
    pipe.add_component("writer", Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="review")))
    pipe.connect("reader.messages", "llm.messages")
    pipe.connect("llm.replies", "writer.replies")
    return pipe


def test_pipeline_run_and_run_async(store):
    s = sid()
    pipe = _pipeline(store)
    first = pipe.run(
        {
            "reader": {"query": "Tell me about Northwind", "session_id": s},
            "writer": {"query": "Tell me about Northwind", "session_id": s},
        }
    )
    assert first["writer"]["messages_written"] == 2

    async def second_turn():
        return await pipe.run_async(
            {
                "reader": {"query": "and Contoso?", "session_id": s},
                "writer": {"query": "and Contoso?", "session_id": s},
            },
            include_outputs_from={"reader"},
        )

    second = asyncio.run(second_turn())
    history = [m.text for m in second["reader"]["history"]]
    assert history == ["Tell me about Northwind", "ANSWER: TELL ME ABOUT NORTHWIND"]
    assert second["writer"]["messages_written"] == 2


def test_pipeline_yaml_round_trip_shares_one_store(store):
    pipe = _pipeline(store)
    yaml_text = pipe.dumps()
    assert PASSWORD not in yaml_text
    assert "NEO4J_PASSWORD" in yaml_text

    # Only the test stand-in generator needs allowlisting; the memory components load by default.
    loaded = Pipeline.loads(yaml_text, allowed_modules=["tests.test_memory_integration"])
    reader, writer = loaded.get_component("reader"), loaded.get_component("writer")
    assert reader.store is writer.store  # one connection after loading
    s = sid()
    out = loaded.run({"reader": {"query": "hello", "session_id": s}, "writer": {"query": "hello", "session_id": s}})
    assert out["writer"]["messages_written"] == 2
    reader.store.close()


def test_reconnect_after_close(store):
    retriever = Neo4jMemoryRetriever(store, recall=RecallConfig(read_long_term=False, read_reasoning=False))
    retriever.run(query="x", session_id=sid())
    store.close()
    assert retriever.run(query="x", session_id=sid())["history"] == []


def test_delete_session(store):
    s = sid()
    Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="skip", assistant="skip")).run(
        query="to be deleted", session_id=s, replies=[ChatMessage.from_assistant("ok")]
    )
    store.delete_session(s)
    retriever = Neo4jMemoryRetriever(store, recall=RecallConfig(read_long_term=False, read_reasoning=False))
    assert retriever.run(query="x", session_id=s)["history"] == []


def test_unreachable_database_degrades_unless_strict(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:1")
    lenient = Neo4jMemoryStore({"extraction": {"extractor_type": "none"}})
    out = Neo4jMemoryRetriever(lenient, system_prompt="sys").run(query="hi", session_id="s")
    assert [m.text for m in out["messages"]] == ["sys", "hi"] and out["context"] == ""
    assert Neo4jMemoryWriter(lenient).run(query="hi", session_id="s") == {
        "messages_written": 0,
        "trace_id": None,
        "entities": [],
    }
    lenient.close()

    strict = Neo4jMemoryStore({"extraction": {"extractor_type": "none"}}, strict=True)
    with pytest.raises(Exception):  # noqa: B017 - the driver's connection error type varies by version
        Neo4jMemoryRetriever(strict).run(query="hi", session_id="s")
    strict.close()


def test_store_registry_is_weak(store):
    data = store.to_dict()
    a = Neo4jMemoryStore.from_dict(data)
    assert Neo4jMemoryStore.from_dict(data) is a
    assert len(store_module._REGISTRY) >= 1


def test_one_read_embeds_the_query_once(store):
    retriever = Neo4jMemoryRetriever(store, recall=OPEN_RECALL)
    retriever.run(query="warm up the connection", session_id=sid())
    embedder = EMBEDDERS[-1]
    before = embedder.calls.copy()
    query = f"a brand new question {uuid.uuid4()}"
    retriever.run(query=query, session_id=sid())
    new_calls = embedder.calls[len(before) :]
    # preferences, entities, facts and traces all search with one shared embedding
    assert new_calls == [query]


# ---------------------------------------------------------------- saving memory directly through the store


def test_store_add_messages_saves_short_term(store, cypher):
    s = sid()
    out = store.add_messages(
        s,
        [
            ChatMessage.from_system("not stored"),
            ChatMessage.from_user("Northwind competes with Contoso"),
            ChatMessage.from_assistant("Noted."),
        ],
        extraction_policy=ExtractionPolicy(user="auto", assistant="skip"),
    )
    assert out["messages_written"] == 2
    assert {"Northwind", "Contoso"} <= {e["name"] for e in out["entities"]}
    rows = cypher(
        "MATCH (:Conversation {session_id: $s})-[:HAS_MESSAGE]->(m:Message) RETURN m.role AS role, m.content AS c "
        "ORDER BY m.timestamp",
        s=s,
    )
    assert rows == [{"role": "user", "c": "Northwind competes with Contoso"}, {"role": "assistant", "c": "Noted."}]
    history = Neo4jMemoryRetriever(store, recall=RecallConfig(read_long_term=False, read_reasoning=False)).run(
        query="next", session_id=s
    )["history"]
    assert [m.text for m in history] == ["Northwind competes with Contoso", "Noted."]


def test_store_add_preference_fact_entity_save_long_term(store, cypher):
    tag = uuid.uuid4().hex[:8]
    pref_id = store.add_preference("region", f"Focus on Germany {tag}", context="investing")
    fact_id = store.add_fact(f"Pravesh-{tag}", "works_at", "Northwind Capital")
    entity = store.add_entity(f"Fabrikam-{tag}", "organization", description="A test company")

    assert cypher("MATCH (p:Preference {id: $id}) RETURN p.category AS c, p.context AS x", id=pref_id) == [
        {"c": "region", "x": "investing"}
    ]
    assert cypher("MATCH (f:Fact {id: $id}) RETURN f.predicate AS p, f.object AS o", id=fact_id) == [
        {"p": "works_at", "o": "Northwind Capital"}
    ]
    assert (entity["name"], entity["type"]) == (f"Fabrikam-{tag}", "ORGANIZATION")
    # Earlier runs left similar "Fabrikam-..." names, so the resolver may flag it; it is never merged into them.
    assert entity["resolution"] in {"none", "flagged"}
    assert cypher("MATCH (e:Entity {id: $id}) RETURN e.description AS d", id=entity["id"]) == [{"d": "A test company"}]

    recalled = Neo4jMemoryRetriever(store, recall=OPEN_RECALL).run(
        query=f"region: Focus on Germany {tag} (investing)", session_id=sid()
    )
    assert any(p["id"] == pref_id for p in recalled["memories"]["preferences"])


def test_store_add_trace_saves_reasoning_including_failures(store, cypher):
    tag = uuid.uuid4().hex[:8]
    ok_call = ToolCall(tool_name="search_companies", arguments={"country": "DE"}, id="c1")
    ok_id = store.add_trace(
        sid(),
        f"Find German AI companies {tag}",
        [
            ChatMessage.from_assistant(tool_calls=[ok_call]),
            ChatMessage.from_tool('[{"name": "Aleph Alpha", "type": "ORGANIZATION"}]', origin=ok_call),
            ChatMessage.from_assistant("Aleph Alpha."),
        ],
    )
    bad_call = ToolCall(tool_name="search_companies", arguments={"country": "FR"}, id="c2")
    bad_id = store.add_trace(
        sid(),
        f"Find French AI companies {tag}",
        [
            ChatMessage.from_assistant(tool_calls=[bad_call]),
            ChatMessage.from_tool("HTTP 503", origin=bad_call, error=True),
        ],
    )
    rows = cypher(
        "MATCH (t:ReasoningTrace) WHERE t.id IN $ids "
        "RETURN t.id AS id, t.success AS ok, t.error_kind AS kind, COUNT { (t)-[:HAS_STEP]->() } AS steps",
        ids=[ok_id, bad_id],
    )
    by_id = {r["id"]: r for r in rows}
    assert (by_id[ok_id]["ok"], by_id[ok_id]["kind"], by_id[ok_id]["steps"]) == (True, None, 1)
    assert (by_id[bad_id]["ok"], by_id[bad_id]["kind"], by_id[bad_id]["steps"]) == (False, "tool_error", 1)


def test_store_save_methods_raise_on_failure(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:1")
    lenient = Neo4jMemoryStore(
        {"extraction": {"extractor_type": "none"}}
    )  # strict=False still raises for explicit saves
    with pytest.raises(Exception):  # noqa: B017 - the driver's connection error type varies by version
        lenient.add_fact("a", "b", "c")
    lenient.close()


# ---------------------------------------------------------------- agent memory tools


def test_agent_tools_store_and_recall_by_user(store, cypher):
    from haystack.components.agents import Agent

    from haystack_integrations.tools.neo4j import Neo4jMemoryRetrieverTool, Neo4jMemoryWriterTool

    from .test_memory_tools import ScriptedChatGenerator

    user = f"user-{uuid.uuid4().hex[:8]}"
    tag = uuid.uuid4().hex[:6]
    generator = ScriptedChatGenerator(
        [
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall(
                        "store_memory",
                        {"memory_type": "preference", "category": "region", "preference": f"Germany {tag}"},
                        id="c1",
                    ),
                    ToolCall(
                        "store_memory",
                        {"memory_type": "entity", "name": f"Northwind-{tag}", "entity_type": "ORGANIZATION"},
                        id="c2",
                    ),
                ]
            ),
            ChatMessage.from_assistant(tool_calls=[ToolCall("retrieve_memories", {}, id="c3")]),
            ChatMessage.from_assistant("Done."),
        ]
    )
    agent = Agent(
        chat_generator=generator,
        tools=[Neo4jMemoryRetrieverTool(memory_store=store), Neo4jMemoryWriterTool(memory_store=store)],
        state_schema={"session_id": {"type": str}, "user_id": {"type": str}},
    )
    result = agent.run(messages=[ChatMessage.from_user("I focus on Germany.")], session_id=sid(), user_id=user)

    results = [r.result for m in result["messages"] for r in m.tool_call_results]
    assert results[:2] == [
        f"Stored preference [region] Germany {tag}.",
        f"Stored entity Northwind-{tag} (ORGANIZATION).",
    ]
    assert f"- [region] Germany {tag}" in results[2]  # recalled for this user, without a query
    linked = cypher("MATCH (:User {identifier: $u})--(p:Preference) RETURN collect(p.preference) AS prefs", u=user)
    assert linked[0]["prefs"] == [f"Germany {tag}"]
