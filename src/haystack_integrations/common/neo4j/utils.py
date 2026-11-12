"""Helpers shared by the Neo4j components."""

from typing import Any

import neo4j


def is_number_list(value: Any) -> bool:
    """True for non-empty lists of numbers, i.e. embedding vectors."""
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def to_json_safe(value: Any) -> Any:
    """Convert Neo4j driver values into JSON-serializable Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "iso_format"):  # neo4j.time.Date / DateTime / Time / Duration
        return value.iso_format()
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]
    if hasattr(value, "items") and callable(value.items):  # neo4j.graph.Node / Relationship
        return {str(k): to_json_safe(v) for k, v in value.items()}
    return str(value)


def patch_cypher_version(driver: neo4j.Driver, version: str) -> None:
    """
    Prefix every query sent through ``driver.execute_query`` with ``CYPHER <version>``.

    Only this driver instance is patched; the ``neo4j.Driver`` class is left untouched.
    """
    original = driver.execute_query

    def execute_query(query: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(query, str) and not query.lstrip().upper().startswith("CYPHER"):
            query = f"CYPHER {version} " + query
        return original(query, *args, **kwargs)

    driver.execute_query = execute_query  # type: ignore[method-assign]
