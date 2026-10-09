"""Assemble final turn evidence without conflating missing cost with zero cost."""

_SESSION_TOKEN_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens", "prompt_tokens", "completion_tokens", "total_tokens",
)
_SESSION_COST_KEYS = ("estimated_cost_usd", "cost_status", "cost_source")


def _usage_fields(agent, api_call_count):
    compressor = agent.context_compressor
    last_prompt_tokens = (
        getattr(compressor, "last_real_prompt_tokens", compressor.last_prompt_tokens)
        if getattr(compressor, "last_prompt_tokens", 0) > 0
        else getattr(compressor, "last_prompt_tokens", 0)
    ) or 0
    return {
        "successful_provider_responses": min(
            api_call_count, getattr(agent, "session_successful_provider_responses", 0)
        ),
        "provider_request_ids": list(getattr(agent, "session_provider_request_ids", [])),
        "usage_telemetry_complete": (
            getattr(agent, "session_usage_missing_responses", 0) == 0
            and getattr(agent, "session_successful_provider_responses", 0) >= api_call_count
        ),
        **{key: getattr(agent, f"session_{key}") for key in _SESSION_TOKEN_KEYS},
        "last_prompt_tokens": last_prompt_tokens,
    }


def _billing_fields(agent):
    fields = {key: getattr(agent, f"session_{key}") for key in _SESSION_COST_KEYS}
    if getattr(agent, "session_cost_estimated_responses", 0):
        fields.update(cost_status="estimated", cost_source="mixed_estimated_responses")
    # Unknown cost takes precedence over estimated: a partial sum is not a run total.
    if getattr(agent, "session_cost_missing_responses", 0):
        fields.update(estimated_cost_usd=None, cost_status="unknown", cost_source="incomplete_responses")
    return fields


def _annotate_result_failures(agent, result, final_response, turn_exit_reason, failed, cleanup_errors):
    if agent._tool_guardrail_halt_decision is not None:
        result["guardrail"] = agent._tool_guardrail_halt_decision.to_metadata()
    if failed and str(turn_exit_reason) == "session_persistence_failed":
        result["error"] = final_response or (
            "session storage could not be written — check the state database "
            "health (`hermes doctor`), then send your message again"
        )
        cause = getattr(agent, "_last_persistence_error_cause", None)
        result["failure_reason"] = "session_persistence_failed:" + (cause or "unknown")
    if cleanup_errors:
        result["cleanup_errors"] = cleanup_errors


def build_turn_result(
    agent, *, final_response, last_reasoning, messages, api_call_count, completed,
    turn_exit_reason, failed, interrupted, response_transformed,
    pre_transform_response, cleanup_errors,
):
    result = {
        "final_response": final_response,
        "last_reasoning": last_reasoning,
        "messages": messages,
        "api_calls": api_call_count,
        **_usage_fields(agent, api_call_count),
        "completed": completed,
        "turn_exit_reason": turn_exit_reason,
        "failed": failed,
        "partial": False,
        "interrupted": interrupted,
        "response_transformed": response_transformed,
        "pre_transform_response": pre_transform_response,
        "response_previewed": getattr(agent, "_response_was_previewed", False),
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        **_billing_fields(agent),
        "service_tier": (
            (getattr(agent, "request_overrides", {}) or {}).get("extra_body") or {}
        ).get("service_tier"),
        "session_id": agent.session_id,
    }
    _annotate_result_failures(agent, result, final_response, turn_exit_reason, failed, cleanup_errors)
    return result
