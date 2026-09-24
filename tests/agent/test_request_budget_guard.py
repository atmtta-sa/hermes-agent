"""Regression contracts for pre-request paid-run budget enforcement."""

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
    max_model_requests_per_run = 8
    session_input_tokens = 0
    session_api_calls = 0

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


def test_stops_before_request_when_cumulative_input_budget_is_exhausted():
    agent = _BudgetAgent()
    agent.session_input_tokens = agent.max_cumulative_input_tokens

    result = prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "continue"}],
        api_call_count=2,
    )

    assert result.action == "stop"


def test_stops_before_request_when_model_request_budget_is_exhausted():
    agent = _BudgetAgent()
    agent.session_api_calls = agent.max_model_requests_per_run

    result = prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "continue"}],
        api_call_count=9,
    )

    assert result.action == "stop"