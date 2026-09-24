"""Test that _build_call_kwargs preserves max_tokens for OpenRouter endpoints.

Regression test for #41035: OpenRouter free-tier credits exhausted when
max_tokens was stripped, causing HTTP 402 and fallback to text-only model.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.auxiliary_client import (
    _LadderRoute,
    _build_call_kwargs,
    _ladder_affordability_rung,
)


class TestOpenRouterMaxTokens:
    """max_tokens must be included for OpenRouter to prevent free-tier 402."""

    def test_openrouter_provider_includes_max_tokens(self):
        """Direct openrouter provider keeps max_tokens."""
        kwargs = _build_call_kwargs(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=2000,
        )
        assert kwargs.get("max_tokens") == 2000 or kwargs.get("max_completion_tokens") == 2000

    def test_openrouter_base_url_includes_max_tokens(self):
        """Custom endpoint with openrouter.ai base_url keeps max_tokens."""
        kwargs = _build_call_kwargs(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=2000,
            base_url="https://openrouter.ai/api/v1",
        )
        assert kwargs.get("max_tokens") == 2000 or kwargs.get("max_completion_tokens") == 2000

    def test_generic_provider_omits_max_tokens(self):
        """Generic OpenAI-compatible provider still omits max_tokens."""
        kwargs = _build_call_kwargs(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=2000,
        )
        assert "max_tokens" not in kwargs

    def test_anthropic_compat_still_includes_max_tokens(self):
        """Anthropic-compatible endpoints still include max_tokens."""
        kwargs = _build_call_kwargs(
            provider="minimax",
            model="MiniMax-Text-01",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=4000,
        )
        assert kwargs["max_tokens"] == 4000

    def test_none_max_tokens_never_included(self):
        """max_tokens=None is never added regardless of provider."""
        for provider, base_url in [
            ("openrouter", None),
            ("openai", "https://openrouter.ai/api/v1"),
            ("minimax", None),
        ]:
            kwargs = _build_call_kwargs(
                provider=provider,
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=None,
                base_url=base_url,
            )
            assert "max_tokens" not in kwargs, (
                f"max_tokens should not be set when None for {provider}"
            )


def test_auxiliary_affordability_error_retries_with_safe_output_cap():
    class BillingError(Exception):
        status_code = 402

    error = BillingError(
        "This request requires more credits, or fewer max_tokens. "
        "You requested up to 131072 tokens, but can only afford 15170."
    )
    route = _LadderRoute(
        client=object(), task="compression", tag="", async_mode=False,
        base_info="https://openrouter.ai/api/v1", resolved_provider="openrouter",
        resolved_model="moonshotai/kimi-k3", resolved_base_url="https://openrouter.ai/api/v1",
        resolved_api_key=None, resolved_api_mode="chat_completions",
        final_model="moonshotai/kimi-k3", main_runtime=None, route_info=None,
    )
    kwargs = {"model": "moonshotai/kimi-k3", "messages": [], "timeout": 30}

    step = next(_ladder_affordability_rung(error, route, kwargs))

    assert step.kind == "call"
    assert step.args[1]["max_tokens"] == 15106


def test_iteration_summary_affordability_error_retries_with_safe_output_cap(monkeypatch):
    from agent import chat_completion_helpers as helpers

    class BillingError(Exception):
        status_code = 402

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        BillingError(
            "This request requires more credits, or fewer max_tokens. "
            "You requested up to 131072 tokens, but can only afford 15170."
        ),
        object(),
    ]
    transport = MagicMock()
    transport.normalize_response.return_value = SimpleNamespace(content="summary")
    agent = SimpleNamespace(
        model="moonshotai/kimi-k3",
        provider="openrouter",
        api_mode="chat_completions",
        _ensure_primary_openai_client=lambda **_kwargs: client,
        _get_transport=lambda: transport,
        _max_tokens_param=lambda value: {"max_tokens": value},
    )
    monkeypatch.setattr(
        helpers,
        "_iteration_summary_chat_kwargs",
        lambda _agent, _messages: {
            "model": "moonshotai/kimi-k3", "messages": [], "max_tokens": 131072,
        },
    )
    monkeypatch.setattr(
        helpers,
        "_managed_summary_call",
        lambda _agent, _request_id, request, callback, **_kwargs: callback(request),
    )

    result = helpers._chat_summary_attempt(agent, [], "request-id")(0)

    assert result == "summary"
    assert client.chat.completions.create.call_args_list[1].kwargs["max_tokens"] == 15106
