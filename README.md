# haystack-neo4j

> This is an experimental Neo4j Labs project and not part of Neo4j's supported product lineup. See the [Neo4j Labs disclaimer](./LABS_DISCLAIMER.txt).

The [Neo4j](https://neo4j.com/) integration for [Haystack](https://haystack.deepset.ai/) 3: one package for
retrieval from existing Neo4j graphs, document storage, and agent memory. Install only the parts you need.

## Components

| Component | Import from | Extra | Status |
| --- | --- | --- | --- |
| `Neo4jGraphRAGRetriever` | `haystack_integrations.components.retrievers.neo4j` | `graphrag` | Available |
| `Neo4jDocumentStore`, `Neo4jEmbeddingRetriever`, `Neo4jDynamicDocumentRetriever`, Cypher query reader/writer | `haystack_integrations.document_stores.neo4j`, `...components.retrievers.neo4j`, `...components.connectors.neo4j` | none (core) | Planned: port of the community [`neo4j-haystack`](https://github.com/prosto/neo4j-haystack) components to Haystack 3 |
| Agent memory store, retriever and writer | `haystack_integrations.memory_stores.neo4j`, `...components.retrievers.neo4j`, `...components.writers.neo4j` | `memory` | Planned: built on [`neo4j-agent-memory`](https://pypi.org/project/neo4j-agent-memory/) |

Every component lives under the `haystack_integrations` namespace, so pipelines that use them save to YAML and load
back with Haystack's default settings.

## Installation

```bash
pip install haystack-neo4j                   # core: haystack-ai + the neo4j driver
pip install "haystack-neo4j[graphrag]"       # + neo4j-graphrag, for Neo4jGraphRAGRetriever
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

## Development

```bash
pip install -e ".[test]"
pytest -m "not integration"                  # unit tests, no network

pip install -e ".[integration]"
pytest -m integration                        # uses the public demo DB neo4j+s://demo.neo4jlabs.com

pip install -e ".[lint]"
ruff check . && ruff format --check .        # lint and formatting (ruff format . to fix)
```

See [`examples/graphrag_retriever.ipynb`](examples/graphrag_retriever.ipynb) for a full walkthrough
with a RAG pipeline and an agent.

## License

MIT