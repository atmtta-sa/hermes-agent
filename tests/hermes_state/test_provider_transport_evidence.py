"""Durable exact-run provider transport evidence contracts."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.provider_transport_evidence import (
    _billing_route_policy,
    finalize_codex_transport,
    record_transport_failure,
    run_with_pretransport_denial,
)
from hermes_state import SessionDB


def _rows(db, sql, params=()):
    with db._lock:
        return [dict(row) for row in db._conn.execute(sql, params).fetchall()]


def test_provider_evidence_schema_is_owned_and_versioned(tmp_path):
    db = SessionDB(tmp_path / "state.db")

    marker = _rows(db, "SELECT value FROM state_meta WHERE key = 'provider_evidence_contract_version'")
    tables = {row["name"] for row in _rows(
        db,
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'provider_%'",
    )}
    attempt_columns = {row["name"] for row in _rows(db, "PRAGMA table_info(provider_transport_attempts)")}
    usage_columns = {row["name"] for row in _rows(db, "PRAGMA table_info(provider_call_usage)")}

    assert marker == [{"value": "3"}]
    assert {"provider_transport_attempts", "provider_call_usage"} <= tables
    assert {
        "route_policy_id", "route_policy_version", "route_policy_digest",
        "credential_principal_id", "billing_mode", "root_chain_request_limit",
        "charge_applicability", "token_accounting_basis",
    } <= attempt_columns
    assert {
        "billing_mode", "charge_applicability", "monetary_currency",
        "token_accounting_basis", "runtime_applicability",
    } <= usage_columns
    usage_info = {row["name"]: row for row in _rows(db, "PRAGMA table_info(provider_call_usage)")}
    assert usage_info["runtime_ms"]["notnull"] == 0


def test_existing_profile_is_upgraded_to_provider_evidence_contract(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    with db._lock:
        assert db._conn is not None
        db._conn.executescript("""
            DROP TABLE provider_call_usage;
            DROP TABLE provider_transport_attempts;
            DELETE FROM state_meta WHERE key = 'provider_evidence_contract_version';
            UPDATE schema_version SET version = 30;
        """)
    db.close()

    upgraded = SessionDB(path)
    marker = _rows(
        upgraded,
        "SELECT value FROM state_meta WHERE key = 'provider_evidence_contract_version'",
    )
    tables = {row["name"] for row in _rows(
        upgraded,
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'provider_%'",
    )}

    assert marker == [{"value": "3"}]
    assert {"provider_transport_attempts", "provider_call_usage"} <= tables


def test_schema_32_usage_survives_nullable_runtime_upgrade(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("session-1", source="tool")
    pricing = json.dumps({"rate_card_id": "a" * 64, "currency": "USD"})
    db.prepare_provider_transport(
        "attempt-1", execution_run_id="run-1", session_id="session-1", sequence=1,
        iteration_attempt=1, provider="custom", model="fixture",
        billing_base_url="https://provider.invalid/v1", pricing_json=pricing,
    )
    db.mark_provider_transport_dispatched("attempt-1")
    db.complete_provider_transport(
        "attempt-1", provider_request_id="request-1", input_tokens=10,
        output_tokens=2, runtime_ms=7, estimated_cost_usd=0.1,
        cost_microusd=100_000, cost_basis="provider_actual",
        cost_authority="provider_usage_response", cost_authority_ref="request-1",
    )
    with db._lock:
        conn = db._conn
        assert conn is not None
        create_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'provider_call_usage'"
        ).fetchone()[0]
        columns = [row[1] for row in conn.execute("PRAGMA table_info(provider_call_usage)")]
        conn.execute("ALTER TABLE provider_call_usage RENAME TO provider_call_usage_current")
        conn.execute(create_sql.replace("runtime_ms INTEGER,", "runtime_ms INTEGER NOT NULL,"))
        names = ", ".join(columns)
        conn.execute(
            f"INSERT INTO provider_call_usage ({names}) SELECT {names} FROM provider_call_usage_current"
        )
        conn.execute("DROP TABLE provider_call_usage_current")
        conn.execute("UPDATE schema_version SET version = 32")
        conn.execute(
            "UPDATE state_meta SET value = '2' WHERE key = 'provider_evidence_contract_version'"
        )
    db.close()

    upgraded = SessionDB(path)
    usage_info = {
        row["name"]: row
        for row in _rows(upgraded, "PRAGMA table_info(provider_call_usage)")
    }
    rows = _rows(
        upgraded, "SELECT provider_request_id, runtime_ms FROM provider_call_usage",
    )

    assert usage_info["runtime_ms"]["notnull"] == 0
    assert rows == [{"provider_request_id": "request-1", "runtime_ms": 7}]


def test_exact_run_records_dispatch_completion_and_pretransport_denial(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-1", source="cli")
    pricing = json.dumps({"rate_card_id": "a" * 64, "currency": "USD"})

    db.prepare_provider_transport(
        "attempt-1", execution_run_id="run-1", session_id="session-1", sequence=1,
        iteration_attempt=1, provider="openai-codex", model="fixture",
        billing_base_url="codex-app-server://local", pricing_json=pricing,
    )
    db.mark_provider_transport_dispatched("attempt-1")
    db.complete_provider_transport(
        "attempt-1", provider_request_id="response-1", input_tokens=100,
        output_tokens=20, runtime_ms=10, estimated_cost_usd=0.0,
        cost_microusd=0, cost_basis="subscription_included",
        cost_authority="openai_codex_authenticated_subscription",
        cost_authority_ref="response-1",
    )
    db.prepare_provider_transport(
        "attempt-2", execution_run_id="run-1", session_id="session-1", sequence=2,
        iteration_attempt=2, provider="openai-codex", model="fixture",
        billing_base_url="codex-app-server://local", pricing_json=pricing,
    )
    db.deny_provider_transport_pretransport("attempt-2", "run_request_budget_exhausted")

    attempts = _rows(
        db,
        "SELECT state, iteration_attempt, dispatched_at, denial_reason "
        "FROM provider_transport_attempts WHERE execution_run_id = ? ORDER BY sequence",
        ("run-1",),
    )
    usage = _rows(db, "SELECT provider_request_id, input_tokens, output_tokens, runtime_ms, "
                  "runtime_basis, estimated_cost_usd, cost_microusd, cost_basis, cost_authority "
                  "FROM provider_call_usage")

    assert attempts == [
        {"state": "completed", "iteration_attempt": 1,
         "dispatched_at": attempts[0]["dispatched_at"], "denial_reason": None},
        {"state": "denied_pretransport", "iteration_attempt": 2,
         "dispatched_at": None, "denial_reason": "run_request_budget_exhausted"},
    ]
    assert attempts[0]["dispatched_at"] is not None
    assert usage == [{"provider_request_id": "response-1", "input_tokens": 100, "output_tokens": 20,
                      "runtime_ms": 10, "runtime_basis": "confirmed_provider_call_ms_v1",
                      "estimated_cost_usd": 0.0, "cost_microusd": 0,
                      "cost_basis": "subscription_included",
                      "cost_authority": "openai_codex_authenticated_subscription"}]


def test_prepare_if_needed_reuses_current_attempt():
    from agent.provider_transport_evidence import prepare_if_needed

    db = Mock()
    agent = SimpleNamespace(
        _provider_transport_attempt_id="attempt-1",
        _session_db=db,
    )

    prepare_if_needed(agent, iteration_attempt=1)

    db.prepare_provider_transport.assert_not_called()


def test_middleware_failure_seals_pretransport_denial():
    db = Mock()
    agent = SimpleNamespace(
        _provider_transport_attempt_id="attempt-1",
        _provider_transport_dispatched=False,
        _session_db=db,
    )

    def fail():
        raise RuntimeError("middleware failed")

    try:
        run_with_pretransport_denial(agent, fail, "middleware_pretransport_denial")
    except RuntimeError:
        pass

    db.deny_provider_transport_pretransport.assert_called_once_with(
        "attempt-1", "middleware_pretransport_denial",
    )


def test_dispatched_failure_remains_unknown():
    db = Mock()
    agent = SimpleNamespace(
        _provider_transport_attempt_id="attempt-1",
        _provider_transport_dispatched=True,
        _session_db=db,
    )

    record_transport_failure(agent, RuntimeError("connection lost"), "pretransport")

    db.mark_provider_transport_unknown.assert_called_once_with("attempt-1", "connection lost")
    db.deny_provider_transport_pretransport.assert_not_called()


def test_uncertain_upstream_crossing_remains_unknown_without_claiming_dispatch():
    db = Mock()
    agent = SimpleNamespace(
        _provider_transport_attempt_id="attempt-1",
        _provider_transport_dispatched=False,
        _provider_transport_uncertain=True,
        _session_db=db,
    )

    record_transport_failure(agent, RuntimeError("app server failed"), "pretransport")

    db.mark_provider_transport_unknown.assert_called_once_with("attempt-1", "app server failed")
    db.mark_provider_transport_dispatched.assert_not_called()
    db.deny_provider_transport_pretransport.assert_not_called()


def test_codex_runtime_failure_after_local_ipc_persists_unknown(tmp_path, monkeypatch):
    from agent import codex_runtime
    from agent.provider_transport_evidence import mark_transport_uncertain

    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-1", source="cli")
    agent = SimpleNamespace(
        _session_db=db,
        _session_db_created=True,
        session_id="session-1",
        session_api_calls=0,
        provider="openai-codex",
        model="fixture",
        base_url="codex-app-server://local",
        _provider_transport_attempt_id=None,
        _provider_transport_dispatched=False,
        _provider_transport_uncertain=False,
    )

    class FailingSession:
        def run_turn(self, **_kwargs):
            mark_transport_uncertain(agent)
            raise RuntimeError("turn/start failed")

    agent._codex_session = FailingSession()
    monkeypatch.setenv("PAPERCLIP_RUN_ID", "run-1")
    monkeypatch.setattr(codex_runtime, "_ensure_codex_session", lambda _agent: None)
    monkeypatch.setattr(codex_runtime, "_close_codex_session", lambda _agent: None)
    monkeypatch.setattr(codex_runtime, "_consume_user_interrupt", lambda *_args: (False, None))

    result = codex_runtime.run_codex_app_server_turn(
        agent, user_message="hi", original_user_message="hi", messages=[],
        effective_task_id="task-1",
    )
    attempts = _rows(
        db,
        "SELECT state, dispatched_at, denial_reason, error_text FROM provider_transport_attempts",
    )

    assert result["completed"] is False
    assert attempts == [{
        "state": "outcome_unknown", "dispatched_at": None,
        "denial_reason": None, "error_text": "turn/start failed",
    }]


def test_codex_completion_remains_unknown_without_upstream_provider_evidence():
    db = Mock()
    agent = SimpleNamespace(
        _provider_transport_attempt_id="attempt-1",
        _provider_transport_dispatched=False,
        _session_db=db,
        provider="openai-codex",
        session_provider_request_ids=[],
    )
    turn = SimpleNamespace(interrupted=False, error=None, turn_id="turn-1")

    finalize_codex_transport(
        agent, turn, {"input_tokens": 100, "output_tokens": 20,
                      "estimated_cost_usd": 0.0, "cost_status": "included"},
    )

    db.mark_provider_transport_unknown.assert_called_once_with(
        "attempt-1", "codex_app_server_upstream_evidence_unavailable",
    )
    db.complete_provider_transport.assert_not_called()
    assert agent.session_provider_request_ids == []


def test_subscription_policy_is_bound_before_transport_and_completes_with_null_money(
    tmp_path, monkeypatch,
):
    from agent.provider_transport_evidence import prepare_transport

    policy = {
        "policyId": "codex-subscription-route", "policyVersion": 1,
        "policyDigest": "b" * 64, "provider": "openai-codex",
        "route": "https://chatgpt.com/backend-api/codex",
        "credentialPrincipalId": "managed-account:codex-uat",
        "modelScope": ["fixture"], "billingMode": "subscription_included",
        "status": "active", "validFrom": "2026-01-01T00:00:00.000Z",
        "validUntil": "2027-01-01T00:00:00.000Z",
        "maxRootChainProviderRequests": 12,
    }
    monkeypatch.setenv("PAPERCLIP_RUN_ID", "run-1")
    monkeypatch.setenv("HERMES_BILLING_ROUTE_POLICY_JSON", json.dumps(policy))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-1", source="tool")
    agent = SimpleNamespace(
        _session_db=db, _session_db_created=True, session_id="session-1",
        provider="openai-codex", model="fixture", base_url="codex-app-server://local",
        _provider_evidence_sequence=0, session_provider_request_ids=[],
    )

    prepare_transport(agent, iteration_attempt=1)
    prepared = _rows(
        db,
        "SELECT contract_version, billing_base_url, route_policy_id, route_policy_version, "
        "route_policy_digest, credential_principal_id, billing_mode, root_chain_request_limit "
        "FROM provider_transport_attempts",
    )
    assert prepared == [{
        "contract_version": 3,
        "billing_base_url": policy["route"],
        "route_policy_id": policy["policyId"],
        "route_policy_version": 1,
        "route_policy_digest": policy["policyDigest"],
        "credential_principal_id": policy["credentialPrincipalId"],
        "billing_mode": "subscription_included",
        "root_chain_request_limit": 12,
    }]

    turn = SimpleNamespace(interrupted=False, error=None, turn_id="turn-1")
    finalize_codex_transport(
        agent, turn, {"input_tokens": 100, "output_tokens": 20,
                      "estimated_cost_usd": 0.0, "cost_status": "included"},
    )
    usage = _rows(
        db,
        "SELECT provider_request_id, runtime_ms, runtime_basis, runtime_applicability, "
        "cost_microusd, cost_basis, cost_authority, "
        "billing_mode, charge_applicability, monetary_currency, token_accounting_basis "
        "FROM provider_call_usage",
    )
    assert usage == [{
        "provider_request_id": "turn-1", "runtime_ms": None,
        "runtime_basis": None, "runtime_applicability": "unavailable_by_route",
        "cost_microusd": None, "cost_basis": "subscription_included",
        "cost_authority": "subscription_route_policy",
        "billing_mode": "subscription_included",
        "charge_applicability": "not_applicable_per_request",
        "monetary_currency": None,
        "token_accounting_basis": "provider_reported_tokens_v1",
    }]
    assert agent.session_provider_request_ids == ["turn-1"]


def _subscription_policy(**overrides):
    policy = {
        "policyId": "codex-subscription-route",
        "policyVersion": 1,
        "policyDigest": "b" * 64,
        "provider": "openai-codex",
        "route": "https://chatgpt.com/backend-api/codex",
        "credentialPrincipalId": "managed-account:codex-uat",
        "modelScope": ["fixture"],
        "billingMode": "subscription_included",
        "status": "active",
        "validFrom": "2026-01-01T00:00:00.000Z",
        "validUntil": "2027-01-01T00:00:00.000Z",
        "maxRootChainProviderRequests": 12,
    }
    policy.update(overrides)
    return policy


@pytest.mark.parametrize(
    "overrides",
    [
        {"route": "https://metered.example/v1"},
        {"modelScope": "fixture"},
        {"policyVersion": True},
        {"maxRootChainProviderRequests": True},
    ],
)
def test_subscription_policy_rejects_route_and_type_drift(monkeypatch, overrides):
    monkeypatch.setenv(
        "HERMES_BILLING_ROUTE_POLICY_JSON",
        json.dumps(_subscription_policy(**overrides)),
    )
    agent = SimpleNamespace(provider="openai-codex", model="fixture")

    with pytest.raises(RuntimeError, match="billing_route_policy_invalid"):
        _billing_route_policy(agent)


def test_cached_subscription_policy_is_revalidated_after_provider_drift(monkeypatch):
    monkeypatch.setenv(
        "HERMES_BILLING_ROUTE_POLICY_JSON",
        json.dumps(_subscription_policy()),
    )
    agent = SimpleNamespace(provider="openai-codex", model="fixture")
    assert _billing_route_policy(agent) is not None

    agent.provider = "openai"

    with pytest.raises(RuntimeError, match="billing_route_policy_invalid"):
        _billing_route_policy(agent)
