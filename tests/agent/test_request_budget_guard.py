"""Regression contracts for pre-request paid-run budget enforcement."""

import json
import time

import agent.conversation_loop as conversation_loop
from agent.agent_init import _load_autonomous_budget_envelope
from agent.conversation_loop import _annotate_request_budget_stop
from agent.turn_iteration_prep import prepare_iteration


class _BudgetAgent:
    step_callback = None
    valid_tool_names = ()
    _skill_nudge_interval = 0
    _interrupt_requested = False
    run_budget_seconds = None
    budget_warning_ratio = None
    iteration_budget = None
    _iteration_budget_warning_injected = False
    _nous_wire_pending = None
    _sanitize_args_cursor = None
    session_id = "continuity-budget-regression"
    logger = None
    max_request_input_tokens = 64_000
    max_cumulative_input_tokens = 200_000
    max_cumulative_output_tokens: int | None = 32_000
    max_model_requests_per_run = 8
    max_run_seconds = 1_800
    max_estimated_cost_usd = 1.0
    max_tokens = 8_000
    session_input_tokens = 0
    session_output_tokens = 0
    session_api_calls = 0
    session_estimated_cost_usd = 0.0
    _run_budget_started_at: float | None = None
    autonomous_budget_required = False
    _request_budget_stop_reason: str | None = None

    def _adopt_nous_key_before_expiry(self):
        return None

    def _drain_pending_steer(self):
        return None

    def _sanitize_tool_call_arguments(self, messages, **kwargs):
        return 0


def test_stops_before_transmitting_request_above_input_ceiling():
    agent = _BudgetAgent()
    messages = [{"role": "user", "content": "token " * 70_000}]

    result = prepare_iteration(agent, messages=messages, api_call_count=1)

    assert result.action == "stop"
    assert agent._request_budget_stop_reason == "prompt_budget_exhausted"


def test_autonomous_prompt_overflow_requests_fresh_session_rollover():
    agent = _BudgetAgent()
    agent.autonomous_budget_required = True
    messages = [{"role": "user", "content": "token " * 70_000}]

    result = prepare_iteration(agent, messages=messages, api_call_count=1)

    assert result.action == "stop"
    assert agent._request_budget_stop_reason == "session_rollover_required"


def test_rollover_result_exits_as_failure_for_paperclip_retry():
    agent = _BudgetAgent()
    agent._request_budget_stop_reason = "session_rollover_required"
    result = {"failed": False, "final_response": "Unfinished work"}

    _annotate_request_budget_stop(agent, result)

    assert result["failed"] is True
    assert result["turn_exit_reason"] == "session_rollover_required"


def test_rollover_result_includes_durable_execution_checkpoint(monkeypatch):
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
    monkeypatch.setattr(
        conversation_loop,
        "build_execution_checkpoint",
        lambda **_kwargs: checkpoint,
        raising=False,
    )
    agent = _BudgetAgent()
    agent._request_budget_stop_reason = "session_rollover_required"
    result = {"failed": False, "final_response": "Unfinished work"}

    _annotate_request_budget_stop(agent, result)

    assert result["execution_checkpoint"] == checkpoint


def test_stops_before_request_when_cumulative_input_budget_is_exhausted():
    agent = _BudgetAgent()
    agent.session_input_tokens = agent.max_cumulative_input_tokens

    result = prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "continue"}],
        api_call_count=2,
    )

    assert result.action == "stop"
    assert agent._request_budget_stop_reason == "run_token_budget_exhausted"


def test_stops_before_request_when_model_request_budget_is_exhausted():
    agent = _BudgetAgent()
    agent.session_api_calls = agent.max_model_requests_per_run

    result = prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "continue"}],
        api_call_count=9,
    )

    assert result.action == "stop"
    assert agent._request_budget_stop_reason == "run_request_budget_exhausted"


def test_stops_before_request_that_would_cross_cumulative_input_budget():
    agent = _BudgetAgent()
    agent.session_input_tokens = 199_999

    result = prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "continue with enough text to consume tokens"}],
        api_call_count=2,
    )

    assert result.action == "stop"
    assert agent._request_budget_stop_reason == "run_token_budget_exhausted"


def test_autonomous_projected_cumulative_input_requests_fresh_session_rollover():
    agent = _BudgetAgent()
    agent.autonomous_budget_required = True
    agent.session_input_tokens = 199_999

    result = prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "continue with enough text to consume tokens"}],
        api_call_count=2,
    )

    assert result.action == "stop"
    assert agent._request_budget_stop_reason == "session_rollover_required"


def test_stops_before_request_when_cumulative_output_budget_is_exhausted():
    agent = _BudgetAgent()
    agent.session_output_tokens = 32_000

    result = prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "continue"}],
        api_call_count=2,
    )

    assert result.action == "stop"
    assert agent._request_budget_stop_reason == "run_token_budget_exhausted"


def test_stops_before_request_when_hard_wall_time_is_exhausted():
    agent = _BudgetAgent()
    agent._run_budget_started_at = time.time() - agent.max_run_seconds - 1

    result = prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "continue"}],
        api_call_count=2,
    )

    assert result.action == "stop"
    assert agent._request_budget_stop_reason == "run_time_budget_exhausted"


def test_paid_autonomous_request_requires_complete_explicit_envelope():
    agent = _BudgetAgent()
    agent.autonomous_budget_required = True
    agent.max_cumulative_output_tokens = None

    result = prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "continue"}],
        api_call_count=1,
    )

    assert result.action == "stop"
    assert agent._request_budget_stop_reason == "autonomous_budget_missing"


def test_stops_before_request_that_would_cross_cost_budget(monkeypatch):
    agent = _BudgetAgent()
    agent.session_estimated_cost_usd = 0.80
    monkeypatch.setattr(
        "agent.turn_iteration_prep._estimate_request_cost_usd",
        lambda _agent, _input_tokens, _output_tokens: 0.25,
        raising=False,
    )

    result = prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "continue"}],
        api_call_count=2,
    )

    assert result.action == "stop"
    assert agent._request_budget_stop_reason == "cost_budget_exhausted"


def test_paid_autonomous_request_stops_when_cost_cannot_be_estimated(monkeypatch):
    agent = _BudgetAgent()
    agent.autonomous_budget_required = True
    monkeypatch.setattr(
        "agent.turn_iteration_prep._estimate_request_cost_usd",
        lambda _agent, _input_tokens, _output_tokens: None,
        raising=False,
    )

    result = prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "continue"}],
        api_call_count=1,
    )

    assert result.action == "stop"
    assert agent._request_budget_stop_reason == "cost_budget_exhausted"


def test_non_paperclip_run_does_not_require_autonomous_envelope():
    required, limits = _load_autonomous_budget_envelope({})

    assert required is False
    assert limits == {}


def test_paperclip_run_without_envelope_fails_closed():
    required, limits = _load_autonomous_budget_envelope({"PAPERCLIP_RUN_ID": "run-1"})

    assert required is True
    assert limits == {}


def test_paperclip_envelope_maps_all_reserved_dimensions():
    required, limits = _load_autonomous_budget_envelope(
        {
            "PAPERCLIP_RUN_ID": "run-1",
            "HERMES_AUTONOMOUS_BUDGET_JSON": json.dumps(
                {
                    "requestCount": 8,
                    "inputTokens": 200_000,
                    "outputTokens": 32_000,
                    "runtimeMs": 900_000,
                    "costMicrousd": 250_000,
                }
            ),
        }
    )

    assert required is True
    assert limits == {
        "max_request_input_tokens": 64_000,
        "max_cumulative_input_tokens": 200_000,
        "max_cumulative_output_tokens": 32_000,
        "max_model_requests_per_run": 8,
        "max_run_seconds": 900.0,
        "max_estimated_cost_usd": 0.25,
        "autonomous_cost_budget_applicable": True,
    }


def test_subscription_envelope_accepts_explicit_non_applicable_cost():
    policy = {
        "policyId": "subscription-policy",
        "policyVersion": 1,
        "policyDigest": "a" * 64,
        "provider": "openai-codex",
        "route": "https://chatgpt.com/backend-api/codex",
        "credentialPrincipalId": "managed-account:test",
        "modelScope": ["gpt-5.6-sol"],
        "billingMode": "subscription_included",
        "status": "active",
        "validFrom": "2026-01-01T00:00:00.000Z",
        "validUntil": "2027-01-01T00:00:00.000Z",
        "maxRootChainProviderRequests": 8,
    }
    required, limits = _load_autonomous_budget_envelope(
        {
            "PAPERCLIP_RUN_ID": "run-subscription",
            "HERMES_AUTONOMOUS_BUDGET_JSON": json.dumps(
                {
                    "requestCount": 4,
                    "inputTokens": 200_000,
                    "outputTokens": 40_000,
                    "runtimeMs": 300_000,
                    "costMicrousd": None,
                }
            ),
            "HERMES_BILLING_ROUTE_POLICY_JSON": json.dumps(policy),
        },
        provider="openai-codex",
        model="gpt-5.6-sol",
    )

    assert required is True
    assert limits == {
        "max_request_input_tokens": 64_000,
        "max_cumulative_input_tokens": 200_000,
        "max_cumulative_output_tokens": 40_000,
        "max_model_requests_per_run": 4,
        "max_run_seconds": 300.0,
        "max_estimated_cost_usd": None,
        "autonomous_cost_budget_applicable": False,
    }


def test_subscription_envelope_rejects_incomplete_route_policy():
    required, limits = _load_autonomous_budget_envelope(
        {
            "PAPERCLIP_RUN_ID": "run-subscription",
            "HERMES_AUTONOMOUS_BUDGET_JSON": json.dumps(
                {
                    "requestCount": 4,
                    "inputTokens": 200_000,
                    "outputTokens": 40_000,
                    "runtimeMs": 300_000,
                    "costMicrousd": None,
                }
            ),
            "HERMES_BILLING_ROUTE_POLICY_JSON": json.dumps(
                {
                    "policyId": "subscription-policy",
                    "policyVersion": 1,
                    "policyDigest": "a" * 64,
                    "credentialPrincipalId": "managed-account:test",
                    "billingMode": "subscription_included",
                    "status": "active",
                    "maxRootChainProviderRequests": 8,
                }
            ),
        },
        provider="openai-codex",
        model="gpt-5.6-sol",
    )

    assert required is True
    assert limits == {}


def test_null_cost_without_subscription_policy_fails_closed():
    required, limits = _load_autonomous_budget_envelope(
        {
            "PAPERCLIP_RUN_ID": "run-metered",
            "HERMES_AUTONOMOUS_BUDGET_JSON": json.dumps(
                {
                    "requestCount": 4,
                    "inputTokens": 200_000,
                    "outputTokens": 40_000,
                    "runtimeMs": 300_000,
                    "costMicrousd": None,
                }
            ),
        }
    )

    assert required is True
    assert limits == {}


def test_malformed_paperclip_envelope_cannot_apply_partial_limits():
    required, limits = _load_autonomous_budget_envelope(
        {
            "PAPERCLIP_RUN_ID": "run-1",
            "HERMES_AUTONOMOUS_BUDGET_JSON": '{"requestCount": 8}',
        }
    )

    assert required is True
    assert limits == {}


def test_paperclip_envelope_rejects_unknown_budget_dimension():
    required, limits = _load_autonomous_budget_envelope(
        {
            "PAPERCLIP_RUN_ID": "run-1",
            "HERMES_AUTONOMOUS_BUDGET_JSON": json.dumps(
                {
                    "requestCount": 8,
                    "inputTokens": 64_000,
                    "outputTokens": 8_000,
                    "runtimeMs": 300_000,
                    "costMicrousd": 250_000,
                    "inputToken": 1,
                }
            ),
        }
    )
    assert required is True
    assert limits == {}
