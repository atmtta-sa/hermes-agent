"""Contracts for provider-free capture at the final physical request boundary."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent.effective_request_capture as effective_request_capture
from agent.effective_request_capture import (
    EffectiveRequestCaptured,
    build_effective_request_artifact,
)


def _capture_into(agent, captured):
    agent._effective_request_capture_sink = lambda capture: captured.append(capture)


def _required_policy(**overrides):
    return {
        "required": True,
        "maxRequestBytes": 1_000_000,
        "maxSystemBytes": 1_000_000,
        "maxMessagesBytes": 1_000_000,
        "maxToolsBytes": 1_000_000,
        "maxToolCount": 100,
        "maxExactDuplicateBlocks": 0,
        "minDuplicateBlockBytes": 512,
        **overrides,
    }


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


def test_chat_policy_denies_oversized_final_request_before_provider_call():
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    client = MagicMock()
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        model="moonshotai/kimi-k3",
        _effective_request_policy=_required_policy(maxRequestBytes=1),
    )
    request = {
        "model": "moonshotai/kimi-k3",
        "messages": [{"role": "user", "content": "bounded task"}],
    }

    with pytest.raises(
        effective_request_capture.EffectiveRequestPolicyDenied,
        match="effective_request_request_bytes_exceeded",
    ):
        _dispatch_nonstreaming_api_request(
            agent,
            request,
            make_client=lambda _reason: client,
        )

    client.chat.completions.create.assert_not_called()
    assert agent._effective_request_manifest["verdict"] == "denied"
    assert agent._effective_request_manifest["reason"] == (
        "effective_request_request_bytes_exceeded"
    )
    assert "request" not in agent._effective_request_manifest


def test_policy_denial_precedes_durable_transport_dispatch():
    from agent.provider_transport_evidence import run_dispatched_transport

    db = MagicMock()
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-sol",
        _effective_request_policy=_required_policy(maxRequestBytes=1),
        _provider_transport_attempt_id="attempt-1",
        _provider_transport_dispatched=False,
        _provider_transport_uncertain=False,
        _session_db=db,
    )
    request = {
        "model": "gpt-5.6-sol",
        "input": [{"role": "user", "content": "bounded task"}],
    }

    with pytest.raises(
        effective_request_capture.EffectiveRequestPolicyDenied,
        match="effective_request_request_bytes_exceeded",
    ):
        run_dispatched_transport(
            agent,
            lambda: effective_request_capture.capture_final_provider_request(
                agent, request, transport="codex_responses",
            ),
        )

    db.mark_provider_transport_dispatched.assert_not_called()
    db.deny_provider_transport_pretransport.assert_called_once_with(
        "attempt-1", "effective_request_request_bytes_exceeded",
    )


def test_codex_policy_denies_duplicate_large_block_before_provider_call():
    from agent.codex_runtime import run_codex_stream
    from run_agent import AIAgent

    repeated = "duplicated contract " * 40
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
    agent._effective_request_policy = _required_policy()
    client = MagicMock()
    request = {
        "model": "gpt-5.6-sol",
        "instructions": repeated,
        "input": [
            {"role": "user", "content": [
                {"type": "input_text", "text": repeated},
            ]},
        ],
        "tools": [],
        "store": False,
    }

    with pytest.raises(
        effective_request_capture.EffectiveRequestPolicyDenied,
        match="effective_request_duplicate_blocks_exceeded",
    ):
        run_codex_stream(agent, request, client=client)

    client.responses.create.assert_not_called()
    assert agent._effective_request_manifest["verdict"] == "denied"
    assert len(agent._effective_request_manifest["duplicate_blocks"]) == 1


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


def test_chat_manifest_is_deterministic_measured_and_content_free():
    repeated = "contract boundary " * 40
    capture = {
        "transport": "chat_completions",
        "provider": "openrouter",
        "model": "moonshotai/kimi-k3",
        "request": {
            "model": "moonshotai/kimi-k3",
            "messages": [
                {"role": "system", "content": repeated},
                {"role": "user", "content": repeated},
            ],
            "tools": [
                {"type": "function", "function": {"name": "terminal"}},
            ],
            "extra_headers": {"Authorization": "Bearer private-token"},
        },
    }

    first = effective_request_capture.build_effective_request_manifest(capture)
    second = effective_request_capture.build_effective_request_manifest(capture)

    assert first == second
    assert first["schema"] == "hermes.effective_request_manifest.v1"
    assert first["transport"] == "chat_completions"
    assert first["provider"] == "openrouter"
    assert first["model"] == "moonshotai/kimi-k3"
    assert first["measurements"]["request_bytes"] > 0
    assert first["measurements"]["system_bytes"] > 0
    assert first["measurements"]["messages_bytes"] > 0
    assert first["measurements"]["tools_bytes"] > 0
    assert first["measurements"]["message_count"] == 2
    assert first["measurements"]["tool_count"] == 1
    assert len(first["hashes"]["request_sha256"]) == 64
    assert first["duplicate_blocks"] == [{
        "sha256": effective_request_capture.hashlib.sha256(
            repeated.encode("utf-8")
        ).hexdigest(),
        "bytes": len(repeated.encode("utf-8")),
        "occurrences": 2,
    }]
    assert "request" not in first
    assert repeated not in json.dumps(first)
    assert "private-token" not in json.dumps(first)


def test_codex_manifest_measures_instructions_input_and_tools_without_content():
    capture = {
        "transport": "codex_responses",
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "request": {
            "model": "gpt-5.6-sol",
            "instructions": "short system instructions",
            "extra_body": {
                "input": [
                    {"role": "user", "content": [
                        {"type": "input_text", "text": "bounded task"},
                    ]},
                ],
                "tools": [
                    {"type": "function", "name": "terminal", "parameters": {}},
                ],
            },
        },
    }

    manifest = effective_request_capture.build_effective_request_manifest(capture)

    assert manifest["measurements"]["system_bytes"] > 0
    assert manifest["measurements"]["messages_bytes"] > 0
    assert manifest["measurements"]["tools_bytes"] > 0
    assert manifest["measurements"]["message_count"] == 1
    assert manifest["measurements"]["tool_count"] == 1
    assert manifest["duplicate_blocks"] == []
    serialized = json.dumps(manifest)
    assert "short system instructions" not in serialized
    assert "bounded task" not in serialized


def test_run_context_telemetry_keeps_every_content_free_request_manifest():
    agent = SimpleNamespace(
        provider="openrouter",
        model="moonshotai/kimi-k3",
        _effective_request_policy=_required_policy(),
    )

    for content in ("first private task", "second private task"):
        effective_request_capture.capture_final_provider_request(
            agent,
            {
                "model": "moonshotai/kimi-k3",
                "messages": [{"role": "user", "content": content}],
                "tools": [],
            },
            transport="chat_completions",
        )

    telemetry = effective_request_capture.build_run_context_telemetry(agent)

    assert telemetry["schema"] == "hermes.context_telemetry.v1"
    assert telemetry["request_count"] == 2
    assert telemetry["verdict"] == "allowed"
    assert len(telemetry["requests"]) == 2
    serialized = json.dumps(telemetry, sort_keys=True)
    assert "first private task" not in serialized
    assert "second private task" not in serialized
    assert "request" not in telemetry["requests"][0]


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
