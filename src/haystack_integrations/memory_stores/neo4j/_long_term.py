"""Long-term memory: recall, the extraction policy, and explicit preferences and facts."""

from __future__ import annotations

import asyncio
from typing import Any

from haystack import logging
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.neo4j import _short_term
from haystack_integrations.memory_stores.neo4j._convert import entities_from_tool_results
from haystack_integrations.memory_stores.neo4j.config import ExtractionPolicy, RecallConfig

logger = logging.getLogger(__name__)

_MESSAGE_ENTITIES = (
    "MATCH (m:Message {id: $id})-[:MENTIONS]->(e:Entity) RETURN DISTINCT e.name AS name, e.type AS type ORDER BY name"
)


def _similarity(item: Any) -> float | None:
    value = (getattr(item, "metadata", None) or {}).get("similarity")
    return float(value) if value is not None else None


# ---------------------------------------------------------------- recall


async def recall(client: Any, query: str, config: RecallConfig) -> dict[str, list[dict[str, Any]]]:
    """Preferences, entities and facts relevant to ``query``, with similarity scores."""
    lt = client.long_term

    async def none() -> list[Any]:
        return []

    preferences, entities, facts = await asyncio.gather(
        lt.search_preferences(query, limit=config.preference_limit, threshold=config.preference_threshold)
        if config.preference_limit
        else none(),
        lt.search_entities(query, limit=config.entity_limit, threshold=config.entity_threshold)
        if config.entity_limit
        else none(),
        lt.search_facts(query, limit=config.fact_limit, threshold=config.fact_threshold)
        if config.include_facts and config.fact_limit
        else none(),
    )
    return {
        "preferences": [
            {
                "id": str(p.id),
                "category": p.category,
                "preference": p.preference,
                "context": p.context,
                "confidence": p.confidence,
                "similarity": _similarity(p),
            }
            for p in preferences
        ],
        "entities": [
            {
                "id": str(e.id),
                "name": e.name,
                "type": str(getattr(e.type, "value", e.type)),
                "description": e.description,
                "similarity": _similarity(e),
            }
            for e in entities
        ],
        "facts": [
            {
                "id": str(f.id),
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence,
                "similarity": _similarity(f),
            }
            for f in facts
        ],
    }


# ---------------------------------------------------------------- explicit writes


def validate_preferences(items: list[dict[str, Any]] | None) -> None:
    for i, item in enumerate(items or []):
        if not isinstance(item, dict) or not item.get("category") or not item.get("preference"):
            msg = f"preferences[{i}] must be a dict with 'category' and 'preference' (optional 'context'): {item!r}"
            raise ValueError(msg)


def validate_facts(items: list[dict[str, Any]] | None) -> None:
    for i, item in enumerate(items or []):
        if not isinstance(item, dict) or not all(item.get(k) for k in ("subject", "predicate", "object")):
            msg = f"facts[{i}] must be a dict with 'subject', 'predicate' and 'object': {item!r}"
            raise ValueError(msg)


def validate_mentions(items: list[dict[str, Any]] | None) -> None:
    for i, item in enumerate(items or []):
        if not isinstance(item, dict) or not (item.get("name") or item.get("id")):
            msg = f"mentions[{i}] must be a dict with 'name' (and usually 'type') or 'id': {item!r}"
            raise ValueError(msg)


async def add_preferences(client: Any, items: list[dict[str, Any]], user_id: str | None) -> int:
    for item in items:
        await client.long_term.add_preference(
            item["category"],
            item["preference"],
            context=item.get("context"),
            confidence=item.get("confidence", 1.0),
            user_identifier=user_id,
        )
    return len(items)


async def add_facts(client: Any, items: list[dict[str, Any]]) -> int:
    for item in items:
        await client.long_term.add_fact(
            item["subject"], item["predicate"], item["object"], confidence=item.get("confidence", 1.0)
        )
    return len(items)


# ---------------------------------------------------------------- extraction policy


def _entity_ref(item: dict[str, Any]) -> Any:
    from neo4j_agent_memory.schema.models import EntityRef  # noqa: PLC0415

    etype = item.get("type")
    return EntityRef(name=item.get("name") or "", type=str(etype).upper() if etype else None, id=item.get("id") or None)


async def _linked_entities(client: Any, message: Any) -> list[dict[str, str]]:
    rows = await client.query.cypher(_MESSAGE_ENTITIES, {"id": str(message.id)})
    return [{"name": r["name"], "type": r["type"]} for r in rows]


def _filter_proposal(proposal: Any, policy: ExtractionPolicy) -> Any:
    allowed = {t.upper() for t in policy.allowed_types} if policy.allowed_types is not None else None
    entities = [
        e
        for e in proposal.entities
        if e.confidence >= policy.min_confidence and (allowed is None or str(e.type).upper() in allowed)
    ]
    relations = [r for r in proposal.relations if r.confidence >= policy.min_confidence]
    return proposal.model_copy(update={"entities": entities, "relations": relations})


async def _write_reviewed(  # noqa: PLR0913, PLR0917
    client: Any,
    session_id: str,
    role: str,
    text: str,
    policy: ExtractionPolicy,
    user_id: str | None,
    strict: bool,
) -> tuple[Any, list[dict[str, str]]]:
    extractor = client.short_term.extractor
    if extractor is None:
        logger.warning("Review extraction needs an extractor, but none is configured; saving without entities.")
        return await _short_term.add_message(client, session_id, role, text, extraction_mode="skip"), []

    try:
        proposal = await extractor.extract(text)
    except Exception as error:
        if strict:
            raise
        logger.warning("Entity extraction failed ({error}); saving the message without entities.", error=error)
        return await _short_term.add_message(
            client, session_id, role, text, extraction_mode="skip", user_id=user_id
        ), []

    proposal = _filter_proposal(proposal, policy)
    if policy.reviewer is not None:
        proposal = policy.reviewer(proposal)

    kept = [{"name": e.name, "type": str(e.type).upper()} for e in proposal.entities]
    message = await _short_term.add_message(
        client,
        session_id,
        role,
        text,
        extraction_mode="explicit",
        mentions=[_entity_ref(e) for e in kept],
        user_id=user_id,
    )

    kept_names = {e["name"] for e in kept}
    if policy.save_relations:
        for relation in proposal.relations:
            if relation.source not in kept_names or relation.target not in kept_names:
                continue
            source = await client.long_term.get_entity_by_name(relation.source)
            target = await client.long_term.get_entity_by_name(relation.target)
            if source is not None and target is not None:
                await client.long_term.add_relationship(
                    source,
                    target,
                    relation.relation_type,
                    confidence=relation.confidence,
                    message_id=str(message.id),
                    extractor="reviewed",
                )
    if policy.save_preferences:
        for preference in proposal.preferences:
            await client.long_term.add_preference(
                preference.category,
                preference.preference,
                context=preference.context,
                confidence=preference.confidence,
                user_identifier=user_id,
            )
    return message, kept


async def write_with_policy(  # noqa: PLR0913
    client: Any,
    *,
    session_id: str,
    role: str,
    text: str,
    mode: str,
    policy: ExtractionPolicy,
    mentions: list[dict[str, Any]] | None = None,
    turn: list[ChatMessage] | None = None,
    user_id: str | None = None,
    strict: bool = False,
) -> tuple[Any, list[dict[str, str]]]:
    """
    Store one message under the extraction ``mode`` for its role, and return it with the entities it linked.

    Each returned entity is ``{"name", "type", "source", "role"}``, where ``source`` names who decided.
    """

    def tag(entities: list[dict[str, str]], source: str) -> list[dict[str, str]]:
        return [{**e, "source": source, "role": role} for e in entities]

    if mode == "skip":
        message = await _short_term.add_message(client, session_id, role, text, extraction_mode="skip", user_id=user_id)
        return message, []

    if mode == "explicit":
        message = await _short_term.add_message(
            client,
            session_id,
            role,
            text,
            extraction_mode="explicit",
            mentions=[_entity_ref(m) for m in mentions or []],
            user_id=user_id,
        )
        return message, tag(await _linked_entities(client, message), "explicit")

    if mode == "tool_results":
        found = entities_from_tool_results(turn or [])
        message = await _short_term.add_message(
            client,
            session_id,
            role,
            text,
            extraction_mode="explicit" if found else "skip",
            mentions=[_entity_ref(e) for e in found],
            user_id=user_id,
        )
        return message, tag(await _linked_entities(client, message), "tool_results") if found else []

    if mode == "review":
        message, kept = await _write_reviewed(client, session_id, role, text, policy, user_id, strict)
        return message, tag(await _linked_entities(client, message), "review") if kept else []

    # auto: the library extracts. The message is stored before extraction runs, so it is never re-sent.
    message = await _short_term.add_message(client, session_id, role, text, extraction_mode="auto", user_id=user_id)
    return message, tag(await _linked_entities(client, message), "extractor")
