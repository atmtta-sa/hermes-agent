"""Bounded terminal-state guard for tool-using ACP turns.

ACP coding agents can emit a polished plan or checkpoint and voluntarily stop
while the latest requested slice still has agent-operable work.  This module
provides a small, surface-specific completion contract.  It is policy-only: it
never executes work and never changes user-visible response text.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


_DEFAULT_MAX_ATTEMPTS = 2
_AGENT_OPERABLE_MAX_ATTEMPTS = 20
_ACTIVE_TODO_STATES = frozenset({"pending", "in_progress"})
_STATUS_RE = re.compile(
    r"<!--\s*HERMES_STATUS:\s*(COMPLETE|BLOCKED)"
    r"(?:\s+reason=\"([^\"]*)\")?\s*-->",
    re.IGNORECASE,
)
_PLACEHOLDER_REASONS = frozenset({"", "...", "tbd", "todo", "unknown", "n/a"})
_AGENT_OPERABLE_BLOCKER_RE = re.compile(
    r"(?:\b(?:background\s+)?(?:process|pytest|test\s+suite|tests?)\b.*"
    r"\b(?:running|active|in\s+progress)\b|"
    r"\b(?:awaiting|waiting\s+for)\b.*\b(?:process|pytest|tests?|suite|result)\b)",
    re.IGNORECASE,
)

ACP_COMPLETION_GUIDANCE = """\
## ACP completion contract

For any ACP turn in which you use tools, end your final response with exactly
one hidden terminal marker:

- `<!-- HERMES_STATUS: COMPLETE -->` only when the latest user request and the
  bounded slice are complete, all active TODOs are resolved, and verification
  required by the task has run successfully.
- `<!-- HERMES_STATUS: BLOCKED reason="..." -->` only when a concrete dependency
  outside the agent's control prevents further safe work; replace `...` with the
  specific dependency.

A plan, design note, status/checkpoint, partial implementation, tests without the
implementation, an agent-operable "next step", or a background process whose
result you can poll or await is not a terminal state. Keep using tools and
continue the current turn instead. Do not expose or explain the marker to the
user.
"""


def _is_real_user_message(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    return not any(
        key.endswith("_synthetic") and value
        for key, value in message.items()
        if isinstance(key, str)
    )


def _current_turn_used_tools(messages: Iterable[dict] | None) -> bool:
    """Return whether an assistant used tools after the latest real user turn."""
    current_turn: list[dict] = []
    for message in messages or []:
        if _is_real_user_message(message):
            current_turn = []
            continue
        if isinstance(message, dict):
            current_turn.append(message)
    return any(
        message.get("role") == "assistant" and bool(message.get("tool_calls"))
        for message in current_turn
    )


def _active_todos(agent: Any) -> list[dict[str, str]]:
    store = getattr(agent, "_todo_store", None)
    if store is None or not callable(getattr(store, "read", None)):
        return []
    try:
        items = store.read()
    except Exception:
        return []
    return [
        item
        for item in items
        if isinstance(item, dict)
        and str(item.get("status") or "").lower() in _ACTIVE_TODO_STATES
    ]


def _format_active_todos(items: list[dict[str, str]]) -> str:
    names = [
        str(item.get("id") or item.get("content") or "unfinished") for item in items
    ]
    shown = ", ".join(f"`{name}`" for name in names[:5])
    if len(names) > 5:
        shown += f", and {len(names) - 5} more"
    return shown


def _terminal_state(final_response: str | None) -> tuple[str, str] | None:
    matches = list(_STATUS_RE.finditer(str(final_response or "")))
    if len(matches) != 1:
        return None
    match = matches[0]
    return match.group(1).upper(), str(match.group(2) or "").strip()


def strip_acp_completion_status(final_response: str | None) -> str:
    """Remove internal ACP terminal markers before returning text to the user."""
    return _STATUS_RE.sub("", str(final_response or "")).rstrip()


def _terminal_state_is_accepted(
    state: tuple[str, str] | None,
    active_todos: list[dict[str, str]],
) -> bool:
    if state is None:
        return False
    status, reason = state
    if status == "COMPLETE":
        return not active_todos
    return (
        status == "BLOCKED"
        and reason.lower() not in _PLACEHOLDER_REASONS
        and not _AGENT_OPERABLE_BLOCKER_RE.search(reason)
    )


def _incomplete_detail(
    state: tuple[str, str] | None,
    active_todos: list[dict[str, str]],
) -> str:
    if active_todos:
        return (
            "You declared completion while active TODO items remain: "
            f"{_format_active_todos(active_todos)}. Resolve or explicitly cancel "
            "them before completing the turn."
        )
    if state is not None and state[0] == "BLOCKED":
        if _AGENT_OPERABLE_BLOCKER_RE.search(state[1]):
            return (
                "The BLOCKED reason describes agent-operable background work. "
                "Poll or wait for the process, inspect its final result, and continue."
            )
        return (
            "The BLOCKED marker does not contain a concrete external dependency. "
            "Replace the placeholder reason or continue the work."
        )
    return (
        "A tool-using ACP turn without a valid COMPLETE or BLOCKED marker is "
        "not a terminal state."
    )


def build_acp_completion_stop_nudge(
    *,
    agent: Any,
    messages: Iterable[dict] | None,
    final_response: str | None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> str | None:
    """Return a bounded continuation nudge for incomplete tool-using ACP turns."""
    if str(getattr(agent, "platform", "") or "").lower() != "acp":
        return None
    if not _current_turn_used_tools(messages):
        return None

    state = _terminal_state(final_response)
    attempt_limit = max_attempts
    if (
        state is not None
        and state[0] == "BLOCKED"
        and _AGENT_OPERABLE_BLOCKER_RE.search(state[1])
    ):
        attempt_limit = max(max_attempts, _AGENT_OPERABLE_MAX_ATTEMPTS)
    if attempts >= attempt_limit:
        return None

    active_todos = _active_todos(agent)
    if _terminal_state_is_accepted(state, active_todos):
        return None
    detail = _incomplete_detail(state, active_todos)

    return (
        "[System: ACP completion contract not satisfied. "
        f"{detail}\n\n"
        "Re-read the latest user request and continue the same bounded slice now. "
        "Do not stop at a plan, checkpoint, partial result, or agent-operable next "
        "step. Finish and verify the work, then use `<!-- HERMES_STATUS: COMPLETE -->`; "
        "or use a BLOCKED marker with a specific non-agent-operable reason.]"
    )


__all__ = [
    "ACP_COMPLETION_GUIDANCE",
    "build_acp_completion_stop_nudge",
    "strip_acp_completion_status",
]
