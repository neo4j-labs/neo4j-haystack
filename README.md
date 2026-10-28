# haystack-neo4j

> This is an experimental Neo4j Labs project and not part of Neo4j's supported product lineup. See the [Neo4j Labs disclaimer](./LABS_DISCLAIMER.txt).

The [Neo4j](https://neo4j.com/) integration for [Haystack](https://haystack.deepset.ai/) 3: one package for
retrieval from existing Neo4j graphs, document storage, and agent memory. Install only the parts you need.

## Components

| Component | Import from | Extra | Status |
| --- | --- | --- | --- |
| `Neo4jGraphRAGRetriever` | `haystack_integrations.components.retrievers.neo4j` | `graphrag` | Available |
| `Neo4jDocumentStore`, `Neo4jEmbeddingRetriever`, `Neo4jDynamicDocumentRetriever`, Cypher query reader/writer | `haystack_integrations.document_stores.neo4j`, `...components.retrievers.neo4j`, `...components.connectors.neo4j` | none (core) | Planned: port of the community [`neo4j-haystack`](https://github.com/prosto/neo4j-haystack) components to Haystack 3 |
| `Neo4jMemoryStore`, `Neo4jMemoryRetriever`, `Neo4jMemoryWriter` (agent memory) | `haystack_integrations.memory_stores.neo4j`, `...components.retrievers.neo4j`, `...components.writers.neo4j` | `memory` | Available, built on [`neo4j-agent-memory`](https://pypi.org/project/neo4j-agent-memory/) |

Every component lives under the `haystack_integrations` namespace, so pipelines that use them save to YAML and load
back with Haystack's default settings.

## Installation

```bash
pip install haystack-neo4j                   # core: haystack-ai + the neo4j driver
pip install "haystack-neo4j[graphrag]"       # + neo4j-graphrag, for Neo4jGraphRAGRetriever
pip install "haystack-neo4j[memory]"         # + neo4j-agent-memory, for the agent memory components
pip install "haystack-neo4j[memory-gliner]"  # memory, plus local GLiNER entity extraction
pip install "haystack-neo4j[all]"            # every optional feature
pip install -e ".[all]"                      # from a clone
```

The package doesn't pick an embedding model for you. Install the Haystack text embedder that matches your index,
for example:

```bash
pip install sentence-transformers-haystack   # local models such as all-MiniLM-L6-v2
# or use haystack's built-in OpenAITextEmbedder / AzureOpenAITextEmbedder
```

## GraphRAG retriever

`Neo4jGraphRAGRetriever` searches an **existing** Neo4j graph and its existing indexes, using Neo4j's official
[`neo4j-graphrag`](https://github.com/neo4j/neo4j-graphrag-python) retrievers. It never writes to the database.
One component covers four search types:

| `search_type` | neo4j-graphrag retriever | What it does |
| --- | --- | --- |
| `vector` | `VectorRetriever` | Vector similarity search on an existing vector index |
| `vector_cypher` | `VectorCypherRetriever` | Vector search, then a graph traversal given by your `retrieval_query` |
| `hybrid` | `HybridRetriever` | Vector + full-text search |
| `hybrid_cypher` | `HybridCypherRetriever` | Hybrid search, then a graph traversal given by your `retrieval_query` |

It works in a `Pipeline` and as a tool for an `Agent` (via `ComponentTool`). Without the `graphrag` extra,
creating it raises an error that names the install command.

### Usage

```python
import os
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.neo4j import Neo4jGraphRAGRetriever

os.environ.setdefault("NEO4J_USERNAME", "companies")
os.environ.setdefault("NEO4J_PASSWORD", "companies")

retriever = Neo4jGraphRAGRetriever(
    url="neo4j+s://demo.neo4jlabs.com",
    database="companies",
    search_type="hybrid_cypher",
    vector_index_name="news_sbert",
    fulltext_index_name="news_fulltext",
    retrieval_query="""
        WITH node AS chunk, score
        MATCH (article:Article)-[:HAS_CHUNK]->(chunk)
        OPTIONAL MATCH (article)-[:MENTIONS]->(org:Organization)
        RETURN chunk.text AS text, article.id AS article_id, article.title AS title,
               collect(DISTINCT org.name)[..5] AS companies, score
        ORDER BY score DESC
    """,
    text_embedder=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
    cypher_version="25",   # the demo DB defaults to Cypher 5
)

for doc in retriever.run(query="Who is investing in electric vehicles?")["documents"]:
    print(doc.score, doc.meta["title"], doc.meta["companies"])

retriever.close()
```

### Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `url` | required | e.g. `bolt://localhost:7687` |
| `username`, `password` | `Secret.from_env_var("NEO4J_USERNAME" / "NEO4J_PASSWORD")` | Stored as env-var references, never plain text |
| `database` | `None` | Neo4j database name |
| `search_type` | `"vector"` | `vector`, `vector_cypher`, `hybrid`, `hybrid_cypher` |
| `vector_index_name` | required | Existing vector index |
| `fulltext_index_name` | `None` | Required for `hybrid*`, rejected otherwise |
| `retrieval_query` | `None` | Required for `*_cypher`, rejected otherwise. Starts from `node` and `score` |
| `return_properties` | `None` | Only for `vector` / `hybrid` |
| `text_field` | `"text"` | Node property or query column that becomes `Document.content` |
| `text_embedder` | `None` | Any Haystack text embedder. Optional if you pass `query_embedding` to `run()` |
| `top_k` | `5` | |
| `cypher_version` | `None` | e.g. `"25"`: prefixes this component's queries with `CYPHER 25` |

`run(query, query_embedding=None, top_k=None, query_params=None)` returns `{"documents": [...]}`.

- `query_params` fill in parameters in your `retrieval_query` (e.g. `$since`). Only for `*_cypher`.
- `retrieval_query` is configuration only and is never a `run()` input. That way, an LLM calling the component
  as a tool can't run arbitrary Cypher.

### Documents

- `vector` / `hybrid`: `content` is `node[text_field]`. `meta` holds the other node properties plus `labels` and
  `element_id`. Embedding vectors (lists of numbers) are removed.
- `vector_cypher` / `hybrid_cypher`: `content` is the `text_field` column. `meta` holds the other columns.
- `score` is the similarity score. Neo4j temporal values become ISO strings, so `meta` is JSON-safe.

### As an agent tool

```python
from haystack.tools import ComponentTool

tool = ComponentTool(
    component=retriever,
    name="news_graph_search",
    description="Search news articles with companies and competitors from the knowledge graph.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for"}},
        "required": ["query"],
    },
)
```

### Serialization

Pipelines that contain the retriever save to YAML (`pipeline.dumps()`) and load back with `Pipeline.loads()`.
No extra configuration is needed, because the component is under the `haystack_integrations` namespace. The
YAML contains only env-var names for credentials.

## Agent memory

Neo4j-backed **short-term, long-term and reasoning memory** around any chat generator, built on
[`neo4j-agent-memory`](https://neo4j.com/labs/agent-memory/). Two components share one store:

```
 query, session_id
        │
        ▼
 ┌──────────────────────┐ messages ┌─────────────────────┐ replies ┌───────────────────┐
 │ Neo4jMemoryRetriever │─────────▶│ OpenAIChatGenerator │────────▶│ Neo4jMemoryWriter │
 └──────────────────────┘          │ (or any generator)  │         └───────────────────┘
            └────────────── Neo4jMemoryStore (one shared connection) ────────┘
```

| Memory | `Neo4jMemoryRetriever` returns | `Neo4jMemoryWriter` stores |
| --- | --- | --- |
| **Short-term** | The session's most recent messages | The query and the answer, as messages in the session |
| **Long-term** | Relevant preferences, entities and facts | Entities and relations under the extraction policy; preferences and facts you pass in |
| **Reasoning** | Similar past tasks with the tools they used, including failed ones and why they failed | A trace per turn, failed turns included: task, tool-call steps and errors, outcome, success, failure reason |

### Quick start

```python
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator

from haystack_integrations.components.retrievers.neo4j import Neo4jMemoryRetriever
from haystack_integrations.components.writers.neo4j import Neo4jMemoryWriter
from haystack_integrations.memory_stores.neo4j import ExtractionPolicy, Neo4jMemoryStore

store = Neo4jMemoryStore()  # NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD; OPENAI_API_KEY for extraction/embeddings

pipe = Pipeline()
pipe.add_component("reader", Neo4jMemoryRetriever(store, system_prompt="You are a research assistant."))
pipe.add_component("llm", OpenAIChatGenerator(model="gpt-5.4-mini"))
pipe.add_component("writer", Neo4jMemoryWriter(store, extraction_policy=ExtractionPolicy(user="review")))
pipe.connect("reader.messages", "llm.messages")
pipe.connect("llm.replies", "writer.replies")

query = "I focus on German AI companies. Which ones should I look at?"
pipe.run({
    "reader": {"query": query, "session_id": "session-1"},
    "writer": {"query": query, "session_id": "session-1"},
})
```

`query` and `session_id` go to both components, because Haystack addresses run data per component. The same
pipeline works with `await pipe.run_async(...)`, in scripts, servers and Jupyter: all memory calls run on one
background event loop owned by the store.

Explicit long-term writes go to the writer:

```python
"writer": {
    "query": query, "session_id": "s1",
    "preferences": [{"category": "region", "preference": "Focus on companies in Germany"}],
    "facts": [{"subject": "Pravesh", "predicate": "works_at", "object": "Northwind Capital"}],
}
```

### Who decides what becomes an entity

Automatic extraction sometimes turns descriptions into entities ("European AI companies" as an
`ORGANIZATION`, "advertising" as an `OBJECT`). `ExtractionPolicy` sets the mode separately for the user's message
and the assistant's answer:

| Mode | What is linked | Who decides |
| --- | --- | --- |
| `auto` | Entities and relations from the extractor (no preferences) | The extractor |
| `review` | Extracted entities that pass `allowed_types`, `min_confidence` and your `reviewer`; relations between kept entities; extracted preferences (`save_preferences`) | The extractor proposes, the policy approves |
| `explicit` | Only the entities in the writer's `mentions` input (`{"name", "type"}` or `{"id"}`) | You |
| `tool_results` (assistant only, default) | Objects with `name` and `type` in the turn's JSON tool results | The tools |
| `skip` | Nothing | Nobody |

Defaults: `user="auto"`, `assistant="tool_results"`. A custom `reviewer` is saved by import path; to load a saved
pipeline that uses one, its module must be on Haystack's deserialization allowlist
(`HAYSTACK_DESERIALIZATION_ALLOWLIST`).

The extractor itself is set on the store: `Neo4jMemoryStore(extraction={"extractor_type": ...})` with `"llm"`
(default here, no model download), `"pipeline"` (spaCy, then GLiNER, then LLM), `"gliner"`, `"spacy"` or `"none"`.

### Retriever outputs and writer inputs

| `Neo4jMemoryRetriever` output | Content |
| --- | --- |
| `messages` | System prompt, a memory system message (only when something was recalled), history, then the query. Connect to a chat generator. |
| `history` | Short-term messages only |
| `context` | Rendered long-term and reasoning recall, for your own prompt template; empty if nothing was found |
| `memories` | `{"preferences", "entities", "facts", "traces"}` with ids and similarity scores |

| `Neo4jMemoryWriter` input | Use |
| --- | --- |
| `query`, `session_id` | Required |
| `replies` | From a chat generator |
| `messages` | A full transcript, e.g. an `Agent`'s `messages`. Only what follows the last user message is written, so injected history is not saved twice |
| `user_id` | Scopes conversations to a user |
| `preferences`, `facts`, `mentions` | Explicit long-term writes; invalid items raise `ValueError` |
| `success` | Outcome for the reasoning trace. Default: `True` when there is an answer and no tool error. Pass a real signal (user feedback, an evaluator) in production |

### Tools and reasoning

Haystack 3 runs tools inside an `Agent`. To record tool calls, use the `Agent` as a plain pipeline component and
send its transcript to the writer:

```python
pipe.add_component("agent", Agent(chat_generator=OpenAIChatGenerator(model="gpt-5.4-mini"), tools=my_tools))
pipe.connect("reader.messages", "agent.messages")
pipe.connect("agent.messages", "writer.messages")
```

`record_reasoning` is `"always"` (default), `"with_tools"` or `"never"`.

Every turn gets a trace, failed ones included, so you can see what went wrong:

| `error_kind` | When |
| --- | --- |
| `tool_error` | A tool returned an error, or a tool call got no result. The tool call keeps its error text. |
| `no_answer` | The turn ended without an answer with text |
| `marked_failed` | You passed `success=False` |

The trace also stores `tool_calls` and `tool_errors` counts. The retriever recalls failed traces too
(`RecallConfig.include_failed_traces`, default `True`), with their `error_kind` and tool errors in `memories["traces"]`,
and lists them in `context` under "Similar past tasks that failed", so the model can avoid repeating the issue. Set
`include_failed_traces=False` to recall only successful tasks.

If a component before the writer raises (for example the LLM call fails), the pipeline stops and the writer doesn't
run. To record that turn too, call the writer yourself: `writer.run(query=query, session_id=session_id, success=False)`
stores the query and a `no_answer` trace.

### Configuration

- `Neo4jMemoryStore(uri, username, password, database, llm_model, embedding_model, extraction, resolution, memory,
  strict)`. Credentials are `Secret`s; `extraction`, `resolution` and `memory` are the library's
  `ExtractionConfig`, `ResolutionConfig` and `MemoryConfig` fields (e.g. `memory={"multi_tenant": True}`). Settings
  are built explicitly, so `NAM_*` variables and `.env` files don't change behaviour.
- `RecallConfig`: what the retriever reads (`read_short_term`, `read_long_term`, `read_reasoning`), limits and
  similarity thresholds (defaults 0.3 preferences, 0.4 entities and facts, 0.5 traces), `include_trace_tools`,
  `include_failed_traces`, and the Jinja2 `template` for `context`.
- Pipelines save to YAML with env-var names only. After `Pipeline.loads`, the retriever and writer share one store
  and connection.

### Errors, privacy and deletion

With `strict=False` (default), memory never breaks a pipeline: if Neo4j is unreachable the retriever returns the
system prompt and query only, and a failed write is logged and reports what was written. With `strict=True`,
errors raise.

Memory stores messages, entities, preferences, facts, and tool arguments and results in traces.
`store.delete_session(session_id)` deletes a session's conversation; long-term memory extracted from it is kept by
design. `store.find_potential_duplicates()` lists entity pairs the resolver flagged.

Limitations: tool durations are not recorded; a step's `thought` is often empty with OpenAI models. Each retriever
run costs one embedding call (the store embeds identical texts once and shares the result across the four memory
searches) and five Neo4j queries plus one per recalled trace.

## Development

```bash
pip install -e ".[test]"
pytest -m "not integration"                  # unit tests, no network

pip install -e ".[integration]"
pytest -m integration                        # uses the public demo DB neo4j+s://demo.neo4jlabs.com

docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5-community
pytest -m neo4j                              # agent memory against a local Neo4j; no API key needed

pip install -e ".[lint]"
ruff check . && ruff format --check .        # lint and formatting (ruff format . to fix)
```

Examples: [`examples/graphrag_retriever.ipynb`](examples/graphrag_retriever.ipynb) (RAG pipeline and agent) and
[`examples/agent_memory.ipynb`](examples/agent_memory.ipynb) (all three memories in a chat pipeline).

## License

MIT