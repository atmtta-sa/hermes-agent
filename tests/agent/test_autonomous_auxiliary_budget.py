"""Autonomous Paperclip runs cannot make unreserved auxiliary provider calls."""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from agent import auxiliary_client


_BUDGET = json.dumps(
    {
        "requestCount": 8,
        "inputTokens": 64_000,
        "outputTokens": 8_000,
        "runtimeMs": 300_000,
        "costMicrousd": 250_000,
    }
)


def _enable_paperclip_budget(monkeypatch):
    monkeypatch.setenv("PAPERCLIP_RUN_ID", "run-1")
    monkeypatch.setenv("HERMES_AUTONOMOUS_BUDGET_JSON", _BUDGET)


def test_sync_auxiliary_call_is_denied_before_provider_dispatch(monkeypatch):
    _enable_paperclip_budget(monkeypatch)
    provider_call = Mock()
    monkeypatch.setattr(auxiliary_client, "_call_llm_impl", provider_call)

    with pytest.raises(RuntimeError, match="autonomous_auxiliary_budget_unreserved"):
        auxiliary_client.call_llm(messages=[{"role": "user", "content": "summarize"}])

    provider_call.assert_not_called()


@pytest.mark.asyncio
async def test_async_auxiliary_call_is_denied_before_provider_dispatch(monkeypatch):
    _enable_paperclip_budget(monkeypatch)
    provider_call = AsyncMock()
    monkeypatch.setattr(auxiliary_client, "_async_call_llm_impl", provider_call)

    with pytest.raises(RuntimeError, match="autonomous_auxiliary_budget_unreserved"):
        await auxiliary_client.async_call_llm(
            messages=[{"role": "user", "content": "summarize"}]
        )

    provider_call.assert_not_awaited()


def test_interactive_auxiliary_call_remains_available(monkeypatch):
    monkeypatch.delenv("PAPERCLIP_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_AUTONOMOUS_BUDGET_JSON", raising=False)
    expected = object()
    provider_call = Mock(return_value=expected)
    monkeypatch.setattr(auxiliary_client, "_call_llm_impl", provider_call)

    result = auxiliary_client.call_llm(messages=[{"role": "user", "content": "summarize"}])

    assert result is expected
    provider_call.assert_called_once()
