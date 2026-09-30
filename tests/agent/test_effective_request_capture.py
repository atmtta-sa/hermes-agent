"""Contracts for provider-free capture at the final physical request boundary."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.effective_request_capture import (
    EffectiveRequestCaptured,
    build_effective_request_artifact,
)


def _capture_into(agent, captured):
    agent._effective_request_capture_sink = lambda capture: captured.append(capture)


def test_codex_capture_receives_final_sdk_kwargs_without_calling_provider():
    from agent.codex_runtime import run_codex_stream
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-5.6-sol",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    agent._interrupt_requested = False
    captured = []
    _capture_into(agent, captured)
    client = MagicMock()
    request = {
        "model": "gpt-5.6-sol",
        "instructions": "system + soul + memory + task",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Ping"}]}
        ],
        "tools": [{"type": "function", "name": "terminal", "parameters": {}}],
        "store": False,
    }

    with pytest.raises(EffectiveRequestCaptured):
        run_codex_stream(agent, request, client=client)

    client.responses.create.assert_not_called()
    assert len(captured) == 1
    assert captured[0]["transport"] == "codex_responses"
    final_request = captured[0]["request"]
    assert final_request["model"] == "gpt-5.6-sol"
    assert final_request["instructions"] == request["instructions"]
    assert final_request["stream"] is True
    assert "input" not in final_request
    assert "tools" not in final_request
    assert final_request["extra_body"]["input"] == request["input"]
    assert final_request["extra_body"]["tools"] == request["tools"]


def test_chat_capture_receives_final_sdk_kwargs_without_calling_provider():
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    captured = []
    client = MagicMock()
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        model="openai/gpt-5.6",
        _effective_request_capture_sink=lambda capture: captured.append(capture),
    )
    request = {
        "model": "openai/gpt-5.6",
        "messages": [
            {"role": "system", "content": "system + soul + memory"},
            {"role": "user", "content": "task"},
        ],
        "tools": [{"type": "function", "function": {"name": "terminal"}}],
        "max_tokens": 4096,
    }

    with pytest.raises(EffectiveRequestCaptured):
        _dispatch_nonstreaming_api_request(
            agent,
            request,
            make_client=lambda _reason: client,
        )

    client.chat.completions.create.assert_not_called()
    assert captured == [
        {
            "transport": "chat_completions",
            "provider": "openrouter",
            "model": "openai/gpt-5.6",
            "request": request,
        }
    ]


def test_streaming_chat_capture_includes_final_stream_fields_without_provider_call():
    from agent.chat_completion_helpers import _StreamingCall

    captured = []
    client = MagicMock()
    agent = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        model="openai/gpt-5.6",
        _effective_request_capture_sink=lambda capture: captured.append(capture),
        _create_request_openai_client=lambda **_kwargs: client,
        _touch_activity=lambda _status: None,
    )
    call = _StreamingCall(agent, {}, None)
    request = {
        "model": "openai/gpt-5.6",
        "messages": [{"role": "user", "content": "task"}],
        "stream": True,
        "timeout": 30.0,
    }

    with pytest.raises(EffectiveRequestCaptured):
        call._open_chat_stream(request)

    client.chat.completions.create.assert_not_called()
    assert len(captured) == 1
    assert captured[0]["request"]["stream"] is True
    assert captured[0]["request"]["timeout"] == 30.0
    assert captured[0]["request"]["stream_options"] == {"include_usage": True}


def test_artifact_is_deterministic_measured_and_redacted():
    capture = {
        "transport": "chat_completions",
        "provider": "openrouter",
        "model": "openai/gpt-5.6",
        "request": {
            "model": "openai/gpt-5.6",
            "messages": [
                {"role": "system", "content": "system + soul + memory"},
                {"role": "user", "content": "task"},
            ],
            "tools": [{"type": "function", "function": {"name": "terminal"}}],
            "extra_headers": {"Authorization": "Bearer private-token"},
        },
    }

    first = build_effective_request_artifact(capture)
    second = build_effective_request_artifact(capture)

    assert first == second
    assert first["schema"] == "hermes.effective_request.v1"
    assert first["request_sha256"] == second["request_sha256"]
    assert first["measurements"]["request_bytes"] > 0
    assert first["measurements"]["messages_bytes"] > 0
    assert first["measurements"]["tools_bytes"] > 0
    assert first["request"]["extra_headers"]["Authorization"] == "[REDACTED]"
    assert "private-token" not in str(first)


def test_renderer_uses_real_context_assembly_and_never_calls_provider(
    tmp_path,
    monkeypatch,
):
    from agent.effective_request_capture import render_effective_request
    from run_agent import AIAgent

    home = tmp_path / "hermes-home"
    memories = home / "memories"
    skill_dir = home / "skills" / "phase-one-skill"
    repository = tmp_path / "repository"
    memories.mkdir(parents=True)
    skill_dir.mkdir(parents=True)
    repository.mkdir()
    (home / "SOUL.md").write_text("SOUL_LAYER_MARKER\n", encoding="utf-8")
    (memories / "MEMORY.md").write_text("MEMORY_LAYER_MARKER\n", encoding="utf-8")
    (memories / "USER.md").write_text("USER_PROFILE_LAYER_MARKER\n", encoding="utf-8")
    (repository / "AGENTS.md").write_text("REPOSITORY_LAYER_MARKER\n", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: phase-one-skill\n"
        "description: SKILL_LAYER_MARKER\n"
        "---\n"
        "# Phase One Skill\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(repository))

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        api_mode="chat_completions",
        model="openai/gpt-5.6",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=False,
        skip_memory=False,
        enabled_toolsets=["file", "skills"],
        ephemeral_system_prompt="PAPERCLIP_PERSONA_LAYER_MARKER",
    )
    provider_client = MagicMock()
    monkeypatch.setattr(
        agent,
        "_create_request_openai_client",
        lambda **_kwargs: provider_client,
    )

    artifact = render_effective_request(
        agent,
        "PAPERCLIP_TASK_LAYER_MARKER\nPAPERCLIP_CONTINUATION_LAYER_MARKER",
    )

    provider_client.chat.completions.create.assert_not_called()
    rendered = json.dumps(artifact["request"], ensure_ascii=False)
    for marker in (
        "SOUL_LAYER_MARKER",
        "MEMORY_LAYER_MARKER",
        "USER_PROFILE_LAYER_MARKER",
        "REPOSITORY_LAYER_MARKER",
        "SKILL_LAYER_MARKER",
        "PAPERCLIP_PERSONA_LAYER_MARKER",
        "PAPERCLIP_TASK_LAYER_MARKER",
        "PAPERCLIP_CONTINUATION_LAYER_MARKER",
    ):
        assert marker in rendered
    assert "read_file" in rendered
    assert artifact["measurements"]["request_bytes"] > 0
