"""Reasoning memory: traces of past tasks, their steps and tool calls, including failures."""

from __future__ import annotations

import asyncio
from typing import Any

from haystack.dataclasses import ChatMessage, ChatRole, ToolCallResult

from haystack_integrations.memory_stores.neo4j._convert import one_line, result_text
from haystack_integrations.memory_stores.neo4j.config import RecallConfig

# Why a trace failed. Stored as ReasoningTrace.error_kind and returned by the memory retriever.
TOOL_ERROR = "tool_error"  # a tool returned an error, or a tool call got no result
NO_ANSWER = "no_answer"  # the turn ended without an answer with text
MARKED_FAILED = "marked_failed"  # the caller passed success=False

NO_RESULT = "No result returned."

# The library stores error_kind (from TraceOutcome) but does not read it back.
_ERROR_KINDS = "MATCH (t:ReasoningTrace) WHERE t.id IN $ids RETURN t.id AS id, t.error_kind AS error_kind"


# ---------------------------------------------------------------- recall


async def recall_traces(client: Any, query: str, config: RecallConfig) -> list[dict[str, Any]]:
    """Similar past tasks, with the tools they used and, for failed ones, why they failed."""
    if config.trace_limit <= 0:
        return []
    traces = await client.reasoning.get_similar_traces(
        query,
        limit=config.trace_limit,
        threshold=config.trace_threshold,
        success_only=not config.include_failed_traces,
    )
    if not traces:
        return []

    async def steps_of(trace: Any) -> tuple[list[str], list[dict[str, str]]]:
        if not config.include_trace_tools:
            return [], []
        # get_similar_traces returns traces without steps.
        full = await client.reasoning.get_trace_with_steps(trace.id)
        names: list[str] = []
        errors: list[dict[str, str]] = []
        for step in getattr(full, "steps", None) or []:
            for call in step.tool_calls or []:
                if call.tool_name not in names:
                    names.append(call.tool_name)
                status = getattr(call, "status", None)
                if str(getattr(status, "value", status)) in {"error", "failure", "timeout"}:
                    detail = getattr(call, "error", None) or getattr(call, "result", None) or ""
                    errors.append({"tool": call.tool_name, "error": one_line(str(detail), 200)})
        return names, errors

    async def error_kinds() -> dict[str, str | None]:
        if all(t.success for t in traces):
            return {}
        rows = await client.query.cypher(_ERROR_KINDS, {"ids": [str(t.id) for t in traces]})
        return {r["id"]: r["error_kind"] for r in rows}

    kinds, *details = await asyncio.gather(error_kinds(), *(steps_of(t) for t in traces))
    return [
        {
            "id": str(t.id),
            "task": t.task,
            "outcome": t.outcome,
            "success": bool(t.success),
            "error_kind": None if t.success else kinds.get(str(t.id)),
            "tools": tool_names,
            "errors": errors,
            "similarity": (t.metadata or {}).get("similarity"),
        }
        for t, (tool_names, errors) in zip(traces, details, strict=True)
    ]


# ---------------------------------------------------------------- record


def has_tool_calls(turn: list[ChatMessage]) -> bool:
    return any(m.tool_calls for m in turn)


def default_success(turn: list[ChatMessage], answer: ChatMessage | None) -> bool:
    """True when there is an answer with text, no tool result reported an error, and every tool call got a result."""
    errors = any(r.error for m in turn for r in m.tool_call_results)
    unanswered = sum(len(m.tool_calls) for m in turn) > sum(len(m.tool_call_results) for m in turn)
    return answer is not None and bool(answer.text) and not errors and not unanswered


def _match_results(turn: list[ChatMessage]) -> tuple[dict[str, ToolCallResult], list[ToolCallResult]]:
    by_id: dict[str, ToolCallResult] = {}
    without_id: list[ToolCallResult] = []
    for message in turn:
        for result in message.tool_call_results:
            if result.origin.id:
                by_id[result.origin.id] = result
            else:
                without_id.append(result)
    return by_id, without_id


def _error_kind(success: bool, tool_errors: int, answer: ChatMessage | None) -> str | None:
    if success:
        return None
    if tool_errors:
        return TOOL_ERROR
    if answer is None or not answer.text:
        return NO_ANSWER
    return MARKED_FAILED


async def record_trace(  # noqa: PLR0913
    client: Any,
    *,
    session_id: str,
    task: str,
    turn: list[ChatMessage],
    answer: ChatMessage | None,
    success: bool,
    user_id: str | None = None,
) -> str:
    """
    Write the whole trace in one go (start, steps, completion), so no trace is left incomplete.

    Every turn is recorded, failed ones included: the outcome stores why it failed (``error_kind``) and how many
    tool calls failed, and each failed tool call keeps its error.
    """
    from neo4j_agent_memory import ToolCallStatus  # noqa: PLC0415
    from neo4j_agent_memory.memory.reasoning import TraceOutcome  # noqa: PLC0415

    rs = client.reasoning
    trace = await rs.start_trace(session_id, task, user_identifier=user_id)
    by_id, without_id = _match_results(turn)
    tool_calls = tool_errors = 0

    for message in turn:
        if not message.is_from(ChatRole.ASSISTANT) or not message.tool_calls:
            continue
        names = [call.tool_name for call in message.tool_calls]
        step = await rs.add_step(trace.id, thought=message.text or None, action=", ".join(names))
        for call in message.tool_calls:
            result = by_id.get(call.id) if call.id else None
            if result is None:
                result = next((r for r in without_id if r.origin.tool_name == call.tool_name), None)
                if result is not None:
                    without_id.remove(result)
            if result is None:
                text, failed, error = None, True, NO_RESULT
            else:
                text = result_text(result)
                failed = result.error
                error = text if failed else None
            tool_calls += 1
            tool_errors += int(failed)
            await rs.record_tool_call(
                step.id,
                call.tool_name,
                call.arguments,
                result=text,
                status=ToolCallStatus.ERROR if failed else ToolCallStatus.SUCCESS,
                error=error,
            )

    summary = one_line(answer.text if answer and answer.text else "No answer.")
    outcome = TraceOutcome(
        success=success,
        summary=summary,
        error_kind=_error_kind(success, tool_errors, answer),
        metrics={"tool_calls": float(tool_calls), "tool_errors": float(tool_errors)},
    )
    await rs.complete_trace(trace.id, outcome=outcome)
    return str(trace.id)
