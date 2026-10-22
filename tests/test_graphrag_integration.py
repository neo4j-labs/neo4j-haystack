"""
Integration tests against the public, read-only Neo4j demo database.

Run with: pytest -m integration
Needs network access and `pip install -e ".[integration]"` (sentence-transformers-haystack for the MiniLM embedder).
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest
from haystack import Document, Pipeline

from haystack_integrations.components.retrievers.neo4j import Neo4jGraphRAGRetriever

pytestmark = pytest.mark.integration

st = pytest.importorskip(
    "haystack_integrations.components.embedders.sentence_transformers",
    reason="sentence-transformers-haystack is needed for integration tests",
)

URL = "neo4j+s://demo.neo4jlabs.com"
DATABASE = "companies"
VECTOR_INDEX = "news_sbert"
FULLTEXT_INDEX = "news_fulltext"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "What are the latest developments in electric vehicles?"

RETRIEVAL_QUERY = """
WITH node AS chunk, score
MATCH (article:Article)-[:HAS_CHUNK]->(chunk)
OPTIONAL MATCH (article)-[:MENTIONS]->(org:Organization)
OPTIONAL MATCH (org)-[:HAS_COMPETITOR]->(rival:Organization)
RETURN chunk.text AS text, article.id AS article_id, article.title AS title,
       toString(article.date) AS date, article.sentiment AS sentiment,
       collect(DISTINCT org.name)[..5] AS companies,
       collect(DISTINCT rival.name)[..5] AS competitors, score
ORDER BY score DESC
"""

RETRIEVAL_QUERY_SINCE = """
WITH node AS chunk, score
MATCH (article:Article)-[:HAS_CHUNK]->(chunk)
WHERE article.date >= datetime($since)
RETURN chunk.text AS text, article.id AS article_id, article.title AS title,
       toString(article.date) AS date, score
ORDER BY score DESC
"""


@pytest.fixture(scope="module", autouse=True)
def demo_credentials():
    os.environ.setdefault("NEO4J_USERNAME", "companies")
    os.environ.setdefault("NEO4J_PASSWORD", "companies")


@pytest.fixture(scope="module")
def embedder():
    emb = st.SentenceTransformersTextEmbedder(model=MODEL)
    emb.warm_up()
    return emb


@pytest.fixture(scope="module")
def query_embedding(embedder):
    return embedder.run(text=QUERY)["embedding"]


def make(search_type, embedder=None, **kwargs):
    params = {
        "url": URL,
        "database": DATABASE,
        "search_type": search_type,
        "vector_index_name": VECTOR_INDEX,
        "text_embedder": embedder,
        "cypher_version": "25",
        "top_k": 5,
    }
    if search_type.startswith("hybrid"):
        params["fulltext_index_name"] = FULLTEXT_INDEX
    if search_type.endswith("_cypher"):
        params["retrieval_query"] = RETRIEVAL_QUERY
    params.update(kwargs)
    return Neo4jGraphRAGRetriever(**params)


@pytest.mark.parametrize("search_type", ["vector", "vector_cypher", "hybrid", "hybrid_cypher"])
def test_all_search_types(search_type, embedder):
    retriever = make(search_type, embedder)
    try:
        docs = retriever.run(query=QUERY)["documents"]
    finally:
        retriever.close()

    assert 0 < len(docs) <= 5
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.content
        assert doc.score is not None
        if search_type.endswith("_cypher"):
            for key in ("article_id", "title", "date", "companies", "competitors"):
                assert key in doc.meta
        else:
            assert "Chunk" in doc.meta["labels"]
            assert doc.meta["element_id"]
            assert not any(k.startswith("embedding") for k in doc.meta)


def test_vector_cypher_matches_neo4j_graphrag(query_embedding):
    import neo4j
    from neo4j_graphrag.retrievers import VectorCypherRetriever

    retriever = make("vector_cypher")
    try:
        docs = retriever.run(query=QUERY, query_embedding=query_embedding)["documents"]
    finally:
        retriever.close()

    driver = neo4j.GraphDatabase.driver(URL, auth=("companies", "companies"))
    try:
        original = driver.execute_query

        def execute_query(query, *args, **kwargs):
            if isinstance(query, str) and not query.lstrip().upper().startswith("CYPHER"):
                query = "CYPHER 25 " + query
            return original(query, *args, **kwargs)

        driver.execute_query = execute_query
        direct = VectorCypherRetriever(
            driver=driver, index_name=VECTOR_INDEX, retrieval_query=RETRIEVAL_QUERY, neo4j_database=DATABASE
        )
        records = direct.get_search_results(query_vector=query_embedding, top_k=5).records
    finally:
        driver.close()

    assert [d.meta["article_id"] for d in docs] == [r["article_id"] for r in records]
    assert [d.score for d in docs] == pytest.approx([r["score"] for r in records])


def test_query_params_date_filter(embedder):
    since = "2023-01-01"
    retriever = make("vector_cypher", embedder, retrieval_query=RETRIEVAL_QUERY_SINCE)
    try:
        docs = retriever.run(query=QUERY, query_params={"since": since})["documents"]
    finally:
        retriever.close()

    assert docs
    for doc in docs:
        assert doc.meta["date"][:10] >= since


def test_pipeline_yaml_round_trip_in_fresh_process(embedder, tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("retriever", make("hybrid_cypher", embedder))
    first = pipeline.run({"retriever": {"query": QUERY}})["retriever"]["documents"]
    assert first
    pipeline.get_component("retriever").close()

    yaml_path = tmp_path / "pipeline.yaml"
    yaml_str = pipeline.dumps()
    assert "companies" in yaml_str  # database name is fine
    assert "NEO4J_PASSWORD" in yaml_str
    yaml_path.write_text(yaml_str)

    script = textwrap.dedent(
        f"""
        import json
        from haystack import Pipeline
        pipeline = Pipeline.loads(open({str(yaml_path)!r}).read())
        docs = pipeline.run({{"retriever": {{"query": {QUERY!r}}}}})["retriever"]["documents"]
        pipeline.get_component("retriever").close()
        print(json.dumps([d.meta["article_id"] for d in docs]))
        """
    )
    env = {**os.environ, "NEO4J_USERNAME": "companies", "NEO4J_PASSWORD": "companies"}
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600, check=False
    )
    assert proc.returncode == 0, proc.stderr
    second_ids = json.loads(proc.stdout.strip().splitlines()[-1])
    assert second_ids == [d.meta["article_id"] for d in first]
