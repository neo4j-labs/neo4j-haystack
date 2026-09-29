# neo4j-graphrag-haystack

A [Haystack](https://haystack.deepset.ai/) 3.0 retriever component for **existing** Neo4j graphs, built on
Neo4j's official [`neo4j-graphrag`](https://github.com/neo4j/neo4j-graphrag-python) retrievers.

`Neo4jGraphRAGRetriever` is one component that covers four search types:

| `search_type` | neo4j-graphrag retriever | What it does |
| --- | --- | --- |
| `vector` | `VectorRetriever` | Vector similarity search on an existing vector index |
| `vector_cypher` | `VectorCypherRetriever` | Vector search, then a graph traversal given by your `retrieval_query` |
| `hybrid` | `HybridRetriever` | Vector + full-text search |
| `hybrid_cypher` | `HybridCypherRetriever` | Hybrid search, then a graph traversal given by your `retrieval_query` |

It works in a `Pipeline` and as a tool for an `Agent` (via `ComponentTool`).

This package is not the same as the PyPI package `neo4j-haystack`, which stores and indexes its own
documents. This one reads from the indexes and graph you already have and returns graph context in each
`Document.meta`.

## Installation

```bash
pip install neo4j-graphrag-haystack          # once published
pip install -e .                             # from a clone
```

The only dependencies are `haystack-ai`, `neo4j-graphrag` and `neo4j`. The package doesn't pick an embedding
model for you. Install the Haystack text embedder that matches your index, for example:

```bash
pip install sentence-transformers-haystack   # local models such as all-MiniLM-L6-v2
# or use haystack's built-in OpenAITextEmbedder / AzureOpenAITextEmbedder
```

## Usage

```python
import os
from haystack.utils import Secret
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.neo4j_graphrag import Neo4jGraphRAGRetriever

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

See [`examples/neo4j_graphrag_component.ipynb`](examples/neo4j_graphrag_component.ipynb) for a full walkthrough
with a RAG pipeline and an agent.

## License

MIT
