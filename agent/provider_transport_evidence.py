"""Managed-run provider transport evidence at the actual transport boundary."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def _run_id() -> str | None:
    value = os.environ.get("PAPERCLIP_RUN_ID", "").strip()
    return value or None


def _pricing_snapshot(agent: Any) -> str:
    route = "|".join(str(value or "") for value in (
        getattr(agent, "provider", ""), getattr(agent, "model", ""),
        getattr(agent, "base_url", ""), "provider-evidence-v1",
    ))
    return json.dumps({
        "rate_card_id": hashlib.sha256(route.encode("utf-8")).hexdigest(),
        "currency": "USD",
    }, sort_keys=True, separators=(",", ":"))


def prepare_transport(agent: Any, *, iteration_attempt: int, retry_attempt: int = 0) -> str | None:
    """Create the exact-run prepared row, or raise before provider transport."""
    run_id = _run_id()
    if run_id is None:
        return None
    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    if db is None or not session_id:
        raise RuntimeError("provider_evidence_store_unavailable")
    if not getattr(agent, "_session_db_created", False):
        agent._ensure_db_session()
    sequence = int(getattr(agent, "_provider_evidence_sequence", 0) or 0) + 1
    agent._provider_evidence_sequence = sequence
    material = f"{run_id}|{session_id}|{sequence}"
    attempt_id = hashlib.sha256(material.encode("utf-8")).hexdigest()
    db.prepare_provider_transport(
        attempt_id, execution_run_id=run_id, session_id=session_id, sequence=sequence,
        iteration_attempt=int(iteration_attempt), provider=str(getattr(agent, "provider", "") or "unknown"),
        model=str(getattr(agent, "model", "") or "unknown"),
        billing_base_url=str(getattr(agent, "base_url", "") or ""),
        pricing_json=_pricing_snapshot(agent),
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


def prepare_if_needed(agent: Any, *, iteration_attempt: int, retry_attempt: int = 0) -> None:
    """Prepare one attempt exactly once across middleware retries."""
    if not getattr(agent, "_provider_transport_attempt_id", None):
        prepare_transport(
            agent, iteration_attempt=iteration_attempt, retry_attempt=retry_attempt,
        )


def deny_pretransport(agent: Any, reason: str) -> None:
    attempt_id = getattr(agent, "_provider_transport_attempt_id", None)
    if attempt_id:
        agent._session_db.deny_provider_transport_pretransport(attempt_id, reason)
        agent._provider_transport_attempt_id = None
        agent._provider_transport_dispatched = False
        agent._provider_transport_uncertain = False


def complete_transport(
    agent: Any, *, provider_request_id: str, input_tokens: int, output_tokens: int,
    runtime_ms: int, cost_usd: float | None, cost_status: str, cost_source: str = "none",
) -> None:
    attempt_id = getattr(agent, "_provider_transport_attempt_id", None)
    if not attempt_id:
        return
    cost_basis = "local_estimate"
    cost_authority = None
    cost_authority_ref = None
    cost_microusd = None
    if cost_status == "actual" and cost_source == "provider_cost_api" and cost_usd is not None:
        cost_basis = "provider_actual"
        cost_authority = "provider_usage_response"
        cost_authority_ref = provider_request_id
        cost_microusd = round(float(cost_usd) * 1_000_000)
    # Local subscription route classification is diagnostic only. Settlement
    # requires a separate authenticated entitlement artifact.
    agent._session_db.complete_provider_transport(
        attempt_id, provider_request_id=provider_request_id, input_tokens=input_tokens,
        output_tokens=output_tokens, runtime_ms=runtime_ms, estimated_cost_usd=cost_usd,
        cost_microusd=cost_microusd, cost_basis=cost_basis,
        cost_authority=cost_authority, cost_authority_ref=cost_authority_ref,
    )
    agent._provider_transport_attempt_id = None
    agent._provider_transport_dispatched = False
    agent._provider_transport_uncertain = False


def mark_unknown(agent: Any, error: BaseException | str) -> None:
    attempt_id = getattr(agent, "_provider_transport_attempt_id", None)
    if attempt_id:
        agent._session_db.mark_provider_transport_unknown(attempt_id, str(error))
        agent._provider_transport_attempt_id = None
        agent._provider_transport_dispatched = False
        agent._provider_transport_uncertain = False


def record_transport_failure(agent: Any, error: BaseException | str, denial_reason: str) -> None:
    """Seal one failed attempt according to whether transport was crossed."""
    if (getattr(agent, "_provider_transport_dispatched", False) or
            getattr(agent, "_provider_transport_uncertain", False)):
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
    agent: Any, operation: Callable[[], T], denial_reason: str,
) -> T:
    """Seal an attempt when orchestration fails before transport dispatch."""
    try:
        return operation()
    except BaseException:
        if (getattr(agent, "_provider_transport_attempt_id", None)
                and not getattr(agent, "_provider_transport_dispatched", False)):
            deny_pretransport(agent, denial_reason)
        raise


def finalize_codex_transport(agent: Any, turn: Any, usage: dict[str, Any]) -> None:
    """Fail closed: app-server completion does not prove upstream provider facts."""
    if turn.interrupted or turn.error is not None or not turn.turn_id:
        record_transport_failure(
            agent, turn.error or "codex_app_server_outcome_unconfirmed",
            "codex_app_server_pretransport_failure",
        )
        return
    if not usage:
        mark_unknown(agent, "codex_app_server_usage_missing")
        return
    mark_unknown(agent, "codex_app_server_upstream_evidence_unavailable")
