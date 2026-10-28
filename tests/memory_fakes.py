"""Deterministic stand-ins for the embedding model and the entity extractor, so memory tests need no API key."""

import hashlib

from neo4j_agent_memory.extraction.base import (
    ExtractedEntity,
    ExtractedPreference,
    ExtractedRelation,
    ExtractionResult,
)

DIMENSIONS = 16


class FakeEmbedder:
    """Hash-based vectors: identical texts score 1.0, unrelated texts score near 0."""

    dimensions = DIMENSIONS
    model = "fake-embedder"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        digest = hashlib.sha256(text.strip().lower().encode()).digest()
        return [b / 127.5 - 1.0 for b in digest[:DIMENSIONS]]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class FakeExtractor:
    """
    Finds a fixed set of names, so tests know exactly what the extractor proposes.

    - "Northwind", "Contoso" -> ORGANIZATION (confidence 0.9); a COMPETES_WITH relation when both appear
    - "Pravesh" -> PERSON (confidence 0.3, so review filters can drop it)
    - "advertising" -> OBJECT (confidence 0.6): a description the review policy should be able to reject
    - a sentence with "prefer" -> one preference
    """

    name = "fake"

    async def extract(self, text, *, entity_types=None, extract_relations=True, extract_preferences=True):
        entities = [
            ExtractedEntity(name=n, type="ORGANIZATION", confidence=0.9) for n in ("Northwind", "Contoso") if n in text
        ]
        if "Pravesh" in text:
            entities.append(ExtractedEntity(name="Pravesh", type="PERSON", confidence=0.3))
        if "advertising" in text:
            entities.append(ExtractedEntity(name="advertising", type="OBJECT", confidence=0.6))
        relations = []
        if "Northwind" in text and "Contoso" in text:
            relations.append(
                ExtractedRelation(source="Northwind", target="Contoso", relation_type="COMPETES_WITH", confidence=0.8)
            )
        preferences = []
        if "prefer" in text.lower():
            preferences.append(ExtractedPreference(category="style", preference=text.strip(), confidence=0.9))
        return ExtractionResult(entities=entities, relations=relations, preferences=preferences, source_text=text)
