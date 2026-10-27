"""Memory configuration: ExtractionPolicy (who decides what becomes an entity) and RecallConfig (what is read)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, get_args

from haystack.utils import deserialize_callable, serialize_callable

UserExtraction = Literal["auto", "review", "explicit", "skip"]
AssistantExtraction = Literal["tool_results", "auto", "review", "explicit", "skip"]

DEFAULT_CONTEXT_TEMPLATE = (
    "## Memory\n"
    "{% if preferences %}User preferences:\n"
    "{% for p in preferences %}- [{{ p.category }}] {{ p.preference }}\n{% endfor %}{% endif %}"
    "{% if entities %}Known entities:\n"
    "{% for e in entities %}- {{ e.name }} ({{ e.type }})\n{% endfor %}{% endif %}"
    "{% if facts %}Facts:\n"
    "{% for f in facts %}- {{ f.subject }} {{ f.predicate }} {{ f.object }}\n{% endfor %}{% endif %}"
    '{% set succeeded = traces | selectattr("success") | list %}'
    "{% if succeeded %}Similar past tasks that succeeded:\n"
    "{% for t in succeeded %}- {{ t.task }} -> {{ t.outcome }}"
    '{% if t.tools %} (tools: {{ t.tools | join(", ") }}){% endif %}\n{% endfor %}{% endif %}'
    '{% set failed = traces | rejectattr("success") | list %}'
    "{% if failed %}Similar past tasks that failed (avoid repeating the issue):\n"
    '{% for t in failed %}- {{ t.task }} -> {{ t.outcome }} [{{ t.error_kind or "failed" }}]'
    "{% for e in t.errors %} {{ e.tool }}: {{ e.error }}{% if not loop.last %};{% endif %}{% endfor %}\n"
    "{% endfor %}{% endif %}"
)


@dataclass
class ExtractionPolicy:
    """
    Who decides what becomes an entity when the memory writer saves a message.

    Modes, per role:

    - ``auto``: the configured extractor decides. Writes entities and relations, but no preferences.
    - ``review``: the extractor proposes; entities are filtered by ``allowed_types`` and ``min_confidence``,
      then passed to ``reviewer``. Approved entities, relations between them and (optionally) extracted
      preferences are written.
    - ``explicit``: only the entities passed in the writer's ``mentions`` input are linked.
    - ``tool_results`` (assistant only): entities named in tool results (JSON objects with ``name`` and
      ``type``) are linked. Nothing is extracted from the answer text.
    - ``skip``: the message is stored without entities.

    :param user: Mode for the user's message.
    :param assistant: Mode for the assistant's answer.
    :param allowed_types: Review mode: entity types to keep (e.g. ``["PERSON", "ORGANIZATION"]``).
        ``None`` keeps every type the extractor returns.
    :param min_confidence: Review mode: drop entities and relations below this confidence.
    :param save_relations: Review mode: write relations whose two endpoints were both kept.
    :param save_preferences: Review mode: write preferences the extractor found.
    :param reviewer: Review mode: a function that receives the filtered ``ExtractionResult`` and returns the
        approved one. Serialized by import path, so it must be a module-level function.
    """

    user: UserExtraction = "auto"
    assistant: AssistantExtraction = "tool_results"
    allowed_types: list[str] | None = None
    min_confidence: float = 0.0
    save_relations: bool = True
    save_preferences: bool = True
    reviewer: Callable[[Any], Any] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.user not in get_args(UserExtraction):
            msg = f"ExtractionPolicy.user must be one of {list(get_args(UserExtraction))}, got {self.user!r}."
            raise ValueError(msg)
        if self.assistant not in get_args(AssistantExtraction):
            msg = (
                f"ExtractionPolicy.assistant must be one of {list(get_args(AssistantExtraction))}, "
                f"got {self.assistant!r}."
            )
            raise ValueError(msg)
        if not 0.0 <= self.min_confidence <= 1.0:
            msg = f"ExtractionPolicy.min_confidence must be between 0 and 1, got {self.min_confidence!r}."
            raise ValueError(msg)
        if self.reviewer is not None and not callable(self.reviewer):
            msg = "ExtractionPolicy.reviewer must be callable."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "reviewer"}
        data["reviewer"] = serialize_callable(self.reviewer) if self.reviewer is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionPolicy:
        data = dict(data)
        if data.get("reviewer"):
            data["reviewer"] = deserialize_callable(data["reviewer"])
        return cls(**data)


@dataclass
class RecallConfig:
    """
    What the memory retriever reads and returns.

    Thresholds are cosine similarities (0-1). The library's own default of 0.7 misses relevant memories with
    common embedding models, so the defaults here are lower.

    ``include_failed_traces`` (default ``True``) also recalls similar tasks that failed, with why they failed
    (``error_kind``) and the tool errors, so the model and the user can see what went wrong before. Set it to
    ``False`` to recall only successful traces.
    """

    read_short_term: bool = True
    read_long_term: bool = True
    read_reasoning: bool = True
    history_limit: int = 10
    preference_limit: int = 5
    preference_threshold: float = 0.3
    entity_limit: int = 5
    entity_threshold: float = 0.4
    include_facts: bool = True
    fact_limit: int = 5
    fact_threshold: float = 0.4
    trace_limit: int = 2
    trace_threshold: float = 0.5
    include_trace_tools: bool = True
    include_failed_traces: bool = True
    template: str = DEFAULT_CONTEXT_TEMPLATE

    def __post_init__(self) -> None:
        for name in ("history_limit", "preference_limit", "entity_limit", "fact_limit", "trace_limit"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                msg = f"RecallConfig.{name} must be a non-negative integer, got {value!r}."
                raise ValueError(msg)
        for name in ("preference_threshold", "entity_threshold", "fact_threshold", "trace_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                msg = f"RecallConfig.{name} must be between 0 and 1, got {value!r}."
                raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecallConfig:
        return cls(**data)
