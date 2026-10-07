import json
from types import SimpleNamespace

import pytest

import cli


@pytest.fixture(autouse=True)
def reset_single_query_finalize_state(monkeypatch):
    monkeypatch.setattr(cli, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli, "_cleanup_done", False)




def test_finalize_single_query_releases_session_when_cleanup_fails(monkeypatch):
    calls = []
    fake_cli = SimpleNamespace(_release_active_session=lambda: calls.append("release"))

    def cleanup(**kwargs):
        calls.append("cleanup")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        cli,
        "_notify_single_query_session_finalize",
        lambda _cli: calls.append("finalize"),
    )
    monkeypatch.setattr(cli, "_run_cleanup", cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        cli._finalize_single_query(fake_cli)

    assert calls == ["finalize", "cleanup", "release"]


def test_finalize_single_query_runs_cleanup_when_finalize_hook_fails(monkeypatch):
    calls = []
    fake_agent = SimpleNamespace(session_id="agent-session", platform="cli")
    fake_cli = SimpleNamespace(
        agent=fake_agent,
        session_id="cli-session",
        _release_active_session=lambda: calls.append("release"),
    )

    def invoke_hook(name, **kwargs):
        calls.append("finalize")
        raise RuntimeError("hook failed")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **kwargs: calls.append("cleanup"))

    cli._finalize_single_query(fake_cli)

    assert calls == ["finalize", "cleanup", "release"]




def test_notify_single_query_session_finalize_uses_agent_session(monkeypatch):
    calls = []
    fake_agent = SimpleNamespace(session_id="agent-session", platform="cli")
    fake_cli = SimpleNamespace(agent=fake_agent, session_id="cli-session")

    def invoke_hook(name, **kwargs):
        calls.append((name, kwargs))

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)

    cli._notify_single_query_session_finalize(fake_cli)

    assert calls == [
        (
            "on_session_finalize",
            {
                "session_id": "agent-session",
                "platform": "cli",
                "reason": "shutdown",
            },
        )
    ]


def test_human_single_query_main_finalizes_after_query(monkeypatch):
    calls = []

    import cli as cli_mod

    class _Console:
        def print(self, *_args, **_kwargs):
            calls.append("query-label")

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = _Console()
            self.session_id = "single-query-session"
            self.agent = SimpleNamespace(
                session_id="single-query-session",
                platform="cli",
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _show_security_advisories(self):
            calls.append("advisories")

        def chat(self, query, images=None):
            calls.append(("chat", query, images))
            return "done"

        def _print_exit_summary(self, clear_screen=True):
            calls.append("summary")

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )

    cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert calls == [
        ("claim", "cli", False),
        "query-label",
        "advisories",
        ("chat", "hello", None),
        "summary",
        ("finalize", "single-query-session"),
    ]


def test_quiet_single_query_main_finalizes_while_preserving_exit_code(monkeypatch):
    calls = []

    import cli as cli_mod

    def run_conversation(*, user_message, conversation_history):
        calls.append(("run", user_message, conversation_history))
        return {
            "final_response": "",
            "error": "provider failed",
            "failed": True,
        }

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.provider = "test-provider"
            self.model = "test-model"
            self.session_id = "quiet-session"
            self.conversation_history = []
            self._active_agent_route_signature = "same-route"
            self.agent = SimpleNamespace(
                session_id="quiet-session",
                platform="cli",
                quiet_mode=False,
                suppress_status_output=False,
                stream_delta_callback=object(),
                tool_gen_callback=object(),
                run_conversation=run_conversation,
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _ensure_runtime_credentials(self):
            calls.append("credentials")
            return True

        def _resolve_turn_agent_config(self, effective_query):
            calls.append(("resolve", effective_query))
            return {
                "signature": "same-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **kwargs):
            calls.append(("init", kwargs))
            return True

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 1
    assert ("claim", "cli", True) in calls
    assert ("run", "hello", []) in calls
    assert calls[-1] == ("finalize", "quiet-session")


def test_quiet_single_query_writes_typed_rollover_result(monkeypatch, tmp_path):
    import cli as cli_mod

    target = tmp_path / "run-result.json"
    monkeypatch.setenv("HERMES_RUN_RESULT_FILE", str(target))
    cli_mod._write_quiet_result_file(
        {
            "failed": True,
            "partial": True,
            "stop_reason": "session_rollover_required",
            "turn_exit_reason": "session_rollover_required",
            "final_response": "must not be persisted",
        },
        "session-before-rollover",
    )

    assert json.loads(target.read_text()) == {
        "failed": True,
        "partial": True,
        "session_id": "session-before-rollover",
        "stop_reason": "session_rollover_required",
        "turn_exit_reason": "session_rollover_required",
        "version": 2,
        "provider": None,
        "model": None,
        "endpoint_class": "unknown",
        "provider_request_ids": [],
        "api_calls": 0,
        "successful_provider_responses": 0,
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "estimated_cost_usd": None,
        "cost_status": None,
        "cost_source": None,
        "usage_telemetry_complete": False,
        "cost_unavailable_reason": "cost_not_reported",
    }


def test_quiet_single_query_persists_execution_checkpoint(monkeypatch, tmp_path):
    import cli as cli_mod

    target = tmp_path / "run-result.json"
    checkpoint = {
        "version": 1,
        "workspace": {
            "cwd": "/workspace",
            "gitHead": "a" * 40,
            "branch": "fix/rollover",
            "statusSha256": "b" * 64,
        },
        "patch": {"kind": "git_diff", "sha256": "c" * 64, "bytes": 12},
        "tests": {"status": "not_run", "commands": []},
        "blockers": {"status": "clear", "evidence": ["managed rollover"]},
        "nextAction": "Continue the current issue from the durable workspace.",
    }
    monkeypatch.setenv("HERMES_RUN_RESULT_FILE", str(target))

    cli_mod._write_quiet_result_file(
        {
            "failed": True,
            "partial": True,
            "stop_reason": "session_rollover_required",
            "turn_exit_reason": "session_rollover_required",
            "execution_checkpoint": checkpoint,
        },
        "session-before-rollover",
    )

    assert json.loads(target.read_text())["execution_checkpoint"] == checkpoint


def test_quiet_result_keeps_usage_and_provider_evidence_without_response_text(monkeypatch, tmp_path):
    import cli as cli_mod

    target = tmp_path / "run-result.json"
    monkeypatch.setenv("HERMES_RUN_RESULT_FILE", str(target))
    cli_mod._write_quiet_result_file({
        "provider": "openrouter", "model": "kimi", "api_calls": 2,
        "successful_provider_responses": 2,
        "input_tokens": 234, "output_tokens": 45,
        "cache_read_tokens": 20, "cache_write_tokens": 0,
        "estimated_cost_usd": 0.001, "cost_status": "estimated",
        "provider_request_ids": ["gen-provider-1"],
        "usage_telemetry_complete": True,
        "final_response": "private task body must not persist",
    }, "session-1")

    report = json.loads(target.read_text())
    assert report["version"] == 2
    assert (report["provider"], report["model"], report["api_calls"]) == ("openrouter", "kimi", 2)
    assert report["successful_provider_responses"] == 2
    assert (report["input_tokens"], report["output_tokens"], report["cache_read_tokens"]) == (234, 45, 20)
    assert report["estimated_cost_usd"] == 0.001
    assert report["usage_telemetry_complete"] is True
    assert report["cost_unavailable_reason"] is None
    assert report["provider_request_ids"] == ["gen-provider-1"]
    assert "private task body" not in target.read_text()


def test_unknown_cost_does_not_become_free(monkeypatch, tmp_path):
    import cli as cli_mod

    target = tmp_path / "run-result.json"
    monkeypatch.setenv("HERMES_RUN_RESULT_FILE", str(target))
    cli_mod._write_quiet_result_file({
        "provider": "openai-codex", "model": "gpt-test", "api_calls": 1,
        "successful_provider_responses": 1, "usage_telemetry_complete": True,
        "input_tokens": 100, "output_tokens": 20,
        "estimated_cost_usd": 0.0, "cost_status": "unknown",
    }, "session-1")

    report = json.loads(target.read_text())
    assert report["estimated_cost_usd"] is None
    assert report["cost_unavailable_reason"] == "cost_not_reported"
    assert (report["input_tokens"], report["output_tokens"]) == (100, 20)


def test_actual_provider_cost_remains_available_in_quiet_result(monkeypatch, tmp_path):
    target = tmp_path / "run-result.json"
    monkeypatch.setenv("HERMES_RUN_RESULT_FILE", str(target))
    cli._write_quiet_result_file({
        "provider": "test", "model": "test-model", "api_calls": 1,
        "successful_provider_responses": 1, "usage_telemetry_complete": True,
        "input_tokens": 10, "output_tokens": 2,
        "estimated_cost_usd": 0.002, "cost_status": "actual",
        "cost_source": "provider_cost_api",
    }, "session-1")
    report = json.loads(target.read_text())
    assert report["estimated_cost_usd"] == 0.002
    assert report["cost_status"] == "actual"
    assert report["cost_source"] == "provider_cost_api"


def test_unsupported_reported_cost_does_not_claim_actual_charge(monkeypatch, tmp_path):
    target = tmp_path / "run-result.json"
    monkeypatch.setenv("HERMES_RUN_RESULT_FILE", str(target))
    cli._write_quiet_result_file({
        "provider": "test", "model": "test-model", "api_calls": 1,
        "successful_provider_responses": 1, "usage_telemetry_complete": True,
        "input_tokens": 10, "output_tokens": 2,
        "estimated_cost_usd": 0.002, "cost_status": "reported",
    }, "session-1")
    report = json.loads(target.read_text())
    assert report["estimated_cost_usd"] is None
    assert report["cost_unavailable_reason"] == "cost_not_reported"


@pytest.mark.parametrize("base_url, expected", [
    ("https://openrouter.ai/api/v1?api_key=synthetic-private", "openrouter_api"),
    ("https://unfamiliar.example/v1?api_key=synthetic-private", "unknown"),
])
def test_quiet_result_classifies_endpoint_without_persisting_url(monkeypatch, tmp_path, base_url, expected):
    target = tmp_path / "run-result.json"
    monkeypatch.setenv("HERMES_RUN_RESULT_FILE", str(target))
    cli._write_quiet_result_file({
        "provider": "custom", "model": "example-model", "api_calls": 1,
        "successful_provider_responses": 1, "usage_telemetry_complete": True,
        "input_tokens": 10, "output_tokens": 2, "base_url": base_url,
    }, "session-1")
    report = json.loads(target.read_text())
    assert report["endpoint_class"] == expected
    assert "synthetic-private" not in target.read_text()
    assert base_url not in target.read_text()
