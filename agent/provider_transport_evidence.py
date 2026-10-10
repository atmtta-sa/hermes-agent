"""Managed-run provider transport evidence at the actual transport boundary."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def _run_id() -> str | None:
    value = os.environ.get("PAPERCLIP_RUN_ID", "").strip()
    return value or None


def _pricing_snapshot(agent: Any) -> str:
    route = "|".join(
        str(value or "")
        for value in (
            getattr(agent, "provider", ""),
            getattr(agent, "model", ""),
            getattr(agent, "base_url", ""),
            "provider-evidence-v1",
        )
    )
    return json.dumps(
        {
            "rate_card_id": hashlib.sha256(route.encode("utf-8")).hexdigest(),
            "currency": "USD",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _valid_policy_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_model_scope(value: Any, model: str) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_nonempty_string(item) for item in value)
        and model in value
    )


def _provider_route(provider: str) -> str:
    from providers import get_provider_profile

    profile = get_provider_profile(provider)
    return str(getattr(profile, "base_url", "") or "").rstrip("/")


def billing_route_policy_is_valid(
    *, provider: str, model: str, policy: dict[str, Any]
) -> bool:
    """Validate a subscription policy against the resolved provider route."""
    try:
        valid_from = datetime.fromisoformat(policy["validFrom"].replace("Z", "+00:00"))
        valid_until = datetime.fromisoformat(policy["validUntil"].replace("Z", "+00:00"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    now = datetime.now(timezone.utc)
    if valid_from.tzinfo is None or valid_until.tzinfo is None:
        return False
    expected_route = _provider_route(provider)
    route = policy.get("route")
    checks = (
        bool(provider),
        bool(model),
        policy.get("billingMode") == "subscription_included",
        policy.get("status") == "active",
        policy.get("provider") == provider,
        _valid_model_scope(policy.get("modelScope"), model),
        _nonempty_string(route),
        bool(expected_route),
        route.rstrip("/") == expected_route if isinstance(route, str) else False,
        _nonempty_string(policy.get("policyId")),
        _positive_integer(policy.get("policyVersion")),
        _valid_policy_digest(policy.get("policyDigest")),
        _nonempty_string(policy.get("credentialPrincipalId")),
        _positive_integer(policy.get("maxRootChainProviderRequests")),
        valid_from <= now < valid_until,
    )
    return all(checks)


def _billing_route_policy(agent: Any) -> dict[str, Any] | None:
    raw = os.environ.get("HERMES_BILLING_ROUTE_POLICY_JSON", "").strip()
    if not raw:
        return None
    try:
        policy = json.loads(raw)
        valid = isinstance(policy, dict) and billing_route_policy_is_valid(
            provider=str(getattr(agent, "provider", "") or ""),
            model=str(getattr(agent, "model", "") or ""),
            policy=policy,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("billing_route_policy_invalid") from exc
    if not valid:
        raise RuntimeError("billing_route_policy_invalid")
    agent._provider_billing_route_policy = policy
    return policy


def _prepared_policy_fields(
    agent: Any, policy: dict[str, Any] | None
) -> dict[str, Any]:
    if policy is None:
        return {
            "billing_base_url": str(getattr(agent, "base_url", "") or ""),
            "contract_version": 2,
        }
    return {
        "billing_base_url": policy["route"],
        "contract_version": 3,
        "route_policy_id": policy["policyId"],
        "route_policy_version": policy["policyVersion"],
        "route_policy_digest": policy["policyDigest"],
        "credential_principal_id": policy["credentialPrincipalId"],
        "billing_mode": policy["billingMode"],
        "root_chain_request_limit": policy["maxRootChainProviderRequests"],
        "charge_applicability": "not_applicable_per_request",
        "token_accounting_basis": "provider_reported_tokens_v1",
    }


def prepare_transport(
    agent: Any, *, iteration_attempt: int, retry_attempt: int = 0
) -> str | None:
    """Create the exact-run prepared row, or raise before provider transport."""
    run_id = _run_id()
    if run_id is None:
        return None
    db, session_id = (
        getattr(agent, "_session_db", None),
        getattr(agent, "session_id", None),
    )
    if db is None or not session_id:
        raise RuntimeError("provider_evidence_store_unavailable")
    if not getattr(agent, "_session_db_created", False):
        agent._ensure_db_session()
    sequence = int(getattr(agent, "_provider_evidence_sequence", 0) or 0) + 1
    agent._provider_evidence_sequence = sequence
    material = f"{run_id}|{session_id}|{sequence}"
    attempt_id = hashlib.sha256(material.encode("utf-8")).hexdigest()
    policy = _billing_route_policy(agent)
    policy_fields = _prepared_policy_fields(agent, policy)
    db.prepare_provider_transport(
        attempt_id,
        execution_run_id=run_id,
        session_id=session_id,
        sequence=sequence,
        iteration_attempt=int(iteration_attempt),
        provider=str(getattr(agent, "provider", "") or "unknown"),
        model=str(getattr(agent, "model", "") or "unknown"),
        pricing_json=_pricing_snapshot(agent),
        **policy_fields,
    )
    agent._provider_transport_attempt_id = attempt_id
    agent._provider_transport_dispatched = False
    agent._provider_transport_uncertain = False
    return attempt_id


def dispatch_transport(agent: Any) -> None:
    attempt_id = getattr(agent, "_provider_transport_attempt_id", None)
    if attempt_id:
        agent._session_db.mark_provider_transport_dispatched(attempt_id)
        agent._provider_transport_dispatched = True


def mark_transport_uncertain(agent: Any) -> None:
    """Record that local IPC began but upstream provider crossing is unknowable."""
    if getattr(agent, "_provider_transport_attempt_id", None):
        agent._provider_transport_uncertain = True


def prepare_if_needed(
    agent: Any, *, iteration_attempt: int, retry_attempt: int = 0
) -> None:
    """Prepare one attempt exactly once across middleware retries."""
    if not getattr(agent, "_provider_transport_attempt_id", None):
        prepare_transport(
            agent,
            iteration_attempt=iteration_attempt,
            retry_attempt=retry_attempt,
        )


def deny_pretransport(agent: Any, reason: str) -> None:
    attempt_id = getattr(agent, "_provider_transport_attempt_id", None)
    if attempt_id:
        agent._session_db.deny_provider_transport_pretransport(attempt_id, reason)
        agent._provider_transport_attempt_id = None
        agent._provider_transport_dispatched = False
        agent._provider_transport_uncertain = False


def complete_transport(
    agent: Any,
    *,
    provider_request_id: str,
    input_tokens: int,
    output_tokens: int,
    runtime_ms: int | None,
    cost_usd: float | None,
    cost_status: str,
    cost_source: str = "none",
) -> None:
    attempt_id = getattr(agent, "_provider_transport_attempt_id", None)
    if not attempt_id:
        return
    policy = _billing_route_policy(agent)
    cost_basis = "local_estimate"
    cost_authority = None
    cost_authority_ref = None
    cost_microusd = None
    billing_mode = None
    charge_applicability = None
    monetary_currency = None
    token_accounting_basis = None
    if policy:
        cost_basis = "subscription_included"
        cost_authority = "subscription_route_policy"
        cost_authority_ref = (
            f"{policy['policyId']}:{policy['policyVersion']}:{policy['policyDigest']}"
        )
        cost_usd = None
        billing_mode = "subscription_included"
        charge_applicability = "not_applicable_per_request"
        token_accounting_basis = "provider_reported_tokens_v1"
    elif (
        cost_status == "actual"
        and cost_source == "provider_cost_api"
        and cost_usd is not None
    ):
        cost_basis = "provider_actual"
        cost_authority = "provider_usage_response"
        cost_authority_ref = provider_request_id
        cost_microusd = round(float(cost_usd) * 1_000_000)
    # Local subscription route classification is diagnostic only. Settlement
    # requires a separate authenticated entitlement artifact.
    agent._session_db.complete_provider_transport(
        attempt_id,
        provider_request_id=provider_request_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        runtime_ms=runtime_ms,
        estimated_cost_usd=cost_usd,
        cost_microusd=cost_microusd,
        cost_basis=cost_basis,
        cost_authority=cost_authority,
        cost_authority_ref=cost_authority_ref,
        billing_mode=billing_mode,
        charge_applicability=charge_applicability,
        monetary_currency=monetary_currency,
        token_accounting_basis=token_accounting_basis,
    )
    agent._provider_transport_attempt_id = None
    agent._provider_transport_dispatched = False
    agent._provider_transport_uncertain = False


def complete_subscription_transport_without_runtime(
    agent: Any,
    *,
    provider_request_id: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None,
    cost_status: str,
    cost_source: str,
) -> bool:
    """Complete a policy-bound subscription response whose route exposes no runtime."""
    if not _billing_route_policy(agent):
        return False
    complete_transport(
        agent,
        provider_request_id=provider_request_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        runtime_ms=None,
        cost_usd=cost_usd,
        cost_status=cost_status,
        cost_source=cost_source,
    )
    return True


def mark_unknown(agent: Any, error: BaseException | str) -> None:
    attempt_id = getattr(agent, "_provider_transport_attempt_id", None)
    if attempt_id:
        agent._session_db.mark_provider_transport_unknown(attempt_id, str(error))
        agent._provider_transport_attempt_id = None
        agent._provider_transport_dispatched = False
        agent._provider_transport_uncertain = False


def record_transport_failure(
    agent: Any, error: BaseException | str, denial_reason: str
) -> None:
    """Seal one failed attempt according to whether transport was crossed."""
    if getattr(agent, "_provider_transport_dispatched", False) or getattr(
        agent, "_provider_transport_uncertain", False
    ):
        mark_unknown(agent, error)
    else:
        deny_pretransport(agent, denial_reason)


def run_dispatched_transport(agent: Any, operation: Callable[[], T]) -> T:
    """Mark dispatch immediately before the real transport and retain uncertainty on failure."""
    dispatch_transport(agent)
    try:
        return operation()
    except BaseException as exc:
        mark_unknown(agent, exc)
        raise


def run_with_pretransport_denial(
    agent: Any,
    operation: Callable[[], T],
    denial_reason: str,
) -> T:
    """Seal an attempt when orchestration fails before transport dispatch."""
    try:
        return operation()
    except BaseException:
        if getattr(agent, "_provider_transport_attempt_id", None) and not getattr(
            agent, "_provider_transport_dispatched", False
        ):
            deny_pretransport(agent, denial_reason)
        raise


def _codex_terminal_usage(turn: Any, usage: dict[str, Any]) -> tuple[int, int] | str:
    if turn.interrupted or turn.error is not None or not turn.turn_id:
        return "codex_app_server_outcome_unconfirmed"
    if not usage:
        return "codex_app_server_usage_missing"
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        not isinstance(input_tokens, int)
        or input_tokens < 0
        or not isinstance(output_tokens, int)
        or output_tokens < 0
    ):
        return "codex_app_server_usage_invalid"
    return input_tokens, output_tokens


def finalize_codex_transport(agent: Any, turn: Any, usage: dict[str, Any]) -> None:
    """Use documented Codex turn identity/tokens; runtime remains unavailable."""
    terminal = _codex_terminal_usage(turn, usage)
    if isinstance(terminal, str):
        if terminal == "codex_app_server_outcome_unconfirmed":
            record_transport_failure(
                agent,
                turn.error or terminal,
                "codex_app_server_pretransport_failure",
            )
        else:
            mark_unknown(agent, terminal)
        return
    if not _billing_route_policy(agent):
        mark_unknown(agent, "codex_app_server_upstream_evidence_unavailable")
        return
    input_tokens, output_tokens = terminal
    if not getattr(agent, "_provider_transport_dispatched", False):
        dispatch_transport(agent)
    complete_transport(
        agent,
        provider_request_id=turn.turn_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        runtime_ms=None,
        cost_usd=None,
        cost_status="included",
        cost_source="none",
    )
    request_ids = getattr(agent, "session_provider_request_ids", None)
    if isinstance(request_ids, list):
        request_ids.append(turn.turn_id)
