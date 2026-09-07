"""Tests for the bounded ACP coding completion-contract guard."""

from __future__ import annotations

from types import SimpleNamespace

from agent.acp_completion_stop import (
    ACP_COMPLETION_GUIDANCE,
    build_acp_completion_stop_nudge,
    strip_acp_completion_status,
)


def _agent(*, platform: str = "acp", todos=None):
    store = SimpleNamespace(read=lambda: list(todos or []))
    return SimpleNamespace(platform=platform, _todo_store=store)


def _tool_turn():
    return [
        {"role": "user", "content": "Continue the approved slice."},
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "read_file",
            "tool_call_id": "call-1",
            "content": "ok",
        },
    ]


def test_guidance_defines_hidden_terminal_states():
    assert "<!-- HERMES_STATUS: COMPLETE -->" in ACP_COMPLETION_GUIDANCE
    assert '<!-- HERMES_STATUS: BLOCKED reason="..." -->' in ACP_COMPLETION_GUIDANCE
    assert "plan" in ACP_COMPLETION_GUIDANCE.lower()
    assert "background process" in ACP_COMPLETION_GUIDANCE.lower()


def test_nudges_tool_using_acp_turn_without_terminal_state():
    response = (
        "Design plan complete; no execution layer was implemented.\n\n"
        "Next recommended step: select whether an operator surface is justified."
    )

    nudge = build_acp_completion_stop_nudge(
        agent=_agent(),
        messages=_tool_turn(),
        final_response=response,
        attempts=0,
    )

    assert nudge is not None
    assert "not a terminal state" in nudge
    assert "continue" in nudge.lower()


def test_complete_is_rejected_while_todos_remain_active():
    nudge = build_acp_completion_stop_nudge(
        agent=_agent(
            todos=[
                {
                    "id": "implement",
                    "content": "Implement the slice",
                    "status": "pending",
                }
            ]
        ),
        messages=_tool_turn(),
        final_response="Done.\n<!-- HERMES_STATUS: COMPLETE -->",
        attempts=0,
    )

    assert nudge is not None
    assert "active TODO" in nudge
    assert "implement" in nudge


def test_complete_is_accepted_when_no_active_todos_remain():
    assert (
        build_acp_completion_stop_nudge(
            agent=_agent(
                todos=[{"id": "verify", "content": "Run tests", "status": "completed"}]
            ),
            messages=_tool_turn(),
            final_response="Verified.\n<!-- HERMES_STATUS: COMPLETE -->",
            attempts=0,
        )
        is None
    )


def test_blocked_requires_a_concrete_reason():
    nudge = build_acp_completion_stop_nudge(
        agent=_agent(),
        messages=_tool_turn(),
        final_response='Blocked.\n<!-- HERMES_STATUS: BLOCKED reason="..." -->',
        attempts=0,
    )

    assert nudge is not None
    assert "concrete" in nudge.lower()


def test_blocked_with_concrete_reason_is_accepted():
    assert (
        build_acp_completion_stop_nudge(
            agent=_agent(),
            messages=_tool_turn(),
            final_response=(
                "Provider credentials must be supplied by the user.\n"
                '<!-- HERMES_STATUS: BLOCKED reason="Missing provider credentials" -->'
            ),
            attempts=0,
        )
        is None
    )


def test_running_background_process_is_not_an_external_blocker():
    nudge = build_acp_completion_stop_nudge(
        agent=_agent(),
        messages=_tool_turn(),
        final_response=(
            "The complete test suite is still running.\n"
            '<!-- HERMES_STATUS: BLOCKED reason="The background pytest process is still running" -->'
        ),
        attempts=0,
    )

    assert nudge is not None
    assert "agent-operable" in nudge


def test_awaiting_test_result_is_not_an_external_blocker():
    nudge = build_acp_completion_stop_nudge(
        agent=_agent(),
        messages=_tool_turn(),
        final_response=(
            "Waiting for the suite result.\n"
            '<!-- HERMES_STATUS: BLOCKED reason="Awaiting the complete test result" -->'
        ),
        attempts=0,
    )

    assert nudge is not None
    assert "agent-operable" in nudge


def test_background_process_uses_extended_but_bounded_nudge_budget():
    response = (
        "The complete test suite is still running.\n"
        '<!-- HERMES_STATUS: BLOCKED reason="The background pytest process is still running" -->'
    )

    assert (
        build_acp_completion_stop_nudge(
            agent=_agent(),
            messages=_tool_turn(),
            final_response=response,
            attempts=2,
        )
        is not None
    )
    assert (
        build_acp_completion_stop_nudge(
            agent=_agent(),
            messages=_tool_turn(),
            final_response=response,
            attempts=20,
        )
        is None
    )


def test_guard_ignores_direct_answers_and_non_acp_surfaces():
    direct_messages = [{"role": "user", "content": "Explain this function."}]
    assert (
        build_acp_completion_stop_nudge(
            agent=_agent(),
            messages=direct_messages,
            final_response="It parses the response.",
            attempts=0,
        )
        is None
    )
    assert (
        build_acp_completion_stop_nudge(
            agent=_agent(platform="webui"),
            messages=_tool_turn(),
            final_response="Inspected it.",
            attempts=0,
        )
        is None
    )


def test_nudge_attempts_are_bounded():
    assert (
        build_acp_completion_stop_nudge(
            agent=_agent(),
            messages=_tool_turn(),
            final_response="Partial status only.",
            attempts=2,
            max_attempts=2,
        )
        is None
    )


def test_terminal_marker_is_removed_from_user_visible_response():
    assert (
        strip_acp_completion_status(
            "Verified and complete.\n<!-- HERMES_STATUS: COMPLETE -->"
        )
        == "Verified and complete."
    )
