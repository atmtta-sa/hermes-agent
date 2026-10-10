"""Per-response usage accounting for the conversation turn loop.

After every successful model API call, ``record_response_usage`` folds ``response.usage``
into: the context compressor (``update_from_response`` + the compression-budget rearm
latch), the usage anchor for display/compression math, per-session token/cost counters,
the state.db token-delta queue, and the observability log line. MoA sessions additionally
fold advisor fan-out usage into the reported counts and price the aggregator at its REAL
model/provider. Logger name stays ``agent.conversation_loop`` for caplog parity.
"""

from __future__ import annotations

import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List

from agent.image_token_cost import calibrate_from_usage
from agent.turn_usage_display import display_usage, log_response_usage
from agent.usage_anchor import capture_usage_anchor, set_usage_anchor
from agent.usage_pricing import estimate_usage_cost, normalize_usage, provider_reported_usage_cost

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class ResponseUsageOutcome:
    """``compression_attempts`` is the (possibly rearmed-to-zero) budget counter;
    ``rearmed`` tells the loop to also clear its preflight-block latch."""

    compression_attempts: int
    rearmed: bool = False


def _loop_mod():
    """Lazy ``agent.conversation_loop`` import (avoids an import cycle)."""
    import agent.conversation_loop as _cl

    return _cl


def _fold_moa_usage(agent, canonical_usage):
    """MoA: fold advisor fan-out usage into REPORTED token counts (only aggregator usage is
    returned, so advisor spend would be invisible) and flush the full-turn trace when
    ``moa.save_traces`` is on. Returns ``(client, canonical_usage, advisor_cost)``."""
    _moa_ref_cost = None
    _moa_client = getattr(agent, "client", None)
    if _moa_client is not None and hasattr(_moa_client, "consume_reference_usage"):
        try:
            _ref_usage, _moa_ref_cost = _moa_client.consume_reference_usage()
            if _ref_usage is not None:
                canonical_usage = canonical_usage + _ref_usage
        except Exception as _moa_acct_exc:  # pragma: no cover - defensive
            logger.debug("MoA reference usage accounting failed: %s", _moa_acct_exc)
    if _moa_client is not None and hasattr(_moa_client, "consume_and_save_trace"):
        try:
            # Streaming path: pass the streamed acting text so the trace is self-contained.
            _agg_streamed_text = getattr(agent, "_current_streamed_assistant_text", "") or ""
            _moa_client.consume_and_save_trace(
                agent.session_id, aggregator_output_fallback=_agg_streamed_text or None
            )
        except Exception as _moa_trace_exc:  # pragma: no cover - defensive
            logger.debug("MoA trace flush failed: %s", _moa_trace_exc)
    return _moa_client, canonical_usage, _moa_ref_cost


def _aggregator_cost(agent, response, aggregator_usage, moa_client):
    """Price at the real MoA aggregator route, preferring a provider-reported charge."""
    model, provider, base_url = agent.model, agent.provider, agent.base_url
    slot = getattr(moa_client, "last_aggregator_slot", None) if moa_client is not None else None
    if slot and slot.get("model"):
        model = slot["model"]
        provider = slot.get("provider") or agent.provider
        base_url = slot.get("base_url") or agent.base_url
    return provider_reported_usage_cost(response.usage, base_url=base_url) or estimate_usage_cost(
        model, aggregator_usage, provider=provider, base_url=base_url,
        api_key=getattr(agent, "api_key", ""),
    )


def _advisor_cost(moa_ref_cost):
    if moa_ref_cost is None:
        return None
    try:
        return float(moa_ref_cost)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _record_response_cost(agent, response, aggregator_usage, moa_client, moa_ref_cost):
    """Price the aggregator; never certify a total containing estimated advisor spend."""
    cost = _aggregator_cost(agent, response, aggregator_usage, moa_client)
    delta = None
    if cost.amount_usd is not None:
        delta = float(cost.amount_usd)
        agent.session_estimated_cost_usd += delta
    advisor_cost = _advisor_cost(moa_ref_cost)
    if advisor_cost is not None:
        agent.session_estimated_cost_usd += advisor_cost
        delta = (delta or 0.0) + advisor_cost
    status = "estimated" if moa_ref_cost is not None and cost.status == "actual" else cost.status
    source = "none" if status != cost.status else cost.source
    agent.session_cost_status = status
    agent.session_cost_source = source
    if cost.amount_usd is None:
        agent.session_cost_missing_responses += 1
    elif status not in {"actual", "included"}:
        agent.session_cost_estimated_responses += 1
    return delta, status, source


def _queue_response_usage(agent, canonical_usage, total_tokens, cost_delta, cost_status, cost_source):
    """Queue the per-call delta without blocking the turn on session DB writes."""
    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.queue_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=canonical_usage.reasoning_tokens,
                estimated_cost_usd=cost_delta,
                cost_status=cost_status,
                cost_source=cost_source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included" if cost_status == "included" else None,
                model=agent.model,
                api_call_count=1,
            )
        except Exception as e:  # silent loss here undercounts analytics
            logger.debug("Token persistence failed (session=%s, tokens=%d): %s",
                         agent.session_id, total_tokens, e)


def _record_missing_usage(agent, compressor, api_duration):
    agent.session_usage_missing_responses += 1
    agent.session_cost_missing_responses += 1
    if getattr(compressor, "awaiting_real_usage_after_compression", False):
        compressor.update_from_response({})
    note_usage_less = getattr(compressor, "note_usage_less_response", None)
    if callable(note_usage_less):
        note_usage_less()
    logger.info(
        "API call #%d: model=%s provider=%s in=? out=? total=? latency=%.1fs usage=unavailable",
        agent.session_api_calls, agent.model, agent.provider or "unknown", api_duration,
    )


def _update_compressor_usage(
    agent, compressor, messages, aggregator_usage, canonical_usage,
    usage_dict, api_call_count, compression_attempts, max_compression_attempts,
):
    prompt_tokens = canonical_usage.prompt_tokens
    completed_compaction_pending = bool(
        getattr(compressor, "_verify_compaction_cleared_threshold", False)
    )
    compressor.update_from_response(usage_dict)
    calibrate_from_usage(agent, messages, aggregator_usage.prompt_tokens)
    new_anchor = capture_usage_anchor(
        aggregator_usage.prompt_tokens, aggregator_usage.output_tokens, messages
    )
    if new_anchor is not None:
        set_usage_anchor(agent, new_anchor, turn_base=api_call_count == 1)
    compression_threshold = int(getattr(compressor, "threshold_tokens", 0) or 0)
    rearmed = _loop_mod()._should_rearm_compression_budget(
        compression_attempts, completed_compaction_pending=completed_compaction_pending,
        prompt_tokens=prompt_tokens, threshold_tokens=compression_threshold,
    )
    if rearmed:
        logger.info(
            "Compression budget rearmed after provider-confirmed "
            "recovery: prompt=%s < threshold=%s (attempts were %s/%s)",
            f"{prompt_tokens:,}", f"{compression_threshold:,}",
            compression_attempts, max_compression_attempts,
        )
        compression_attempts = 0
    return compression_attempts, rearmed


def _persist_confirmed_context(agent, compressor):
    if getattr(compressor, "_context_probed", False):
        ctx = compressor.context_length
        if getattr(compressor, "_context_probe_persistable", False):
            from agent.model_metadata import save_context_length

            save_context_length(agent.model, agent.base_url, ctx)
            agent._safe_print(f"{agent.log_prefix}💾 Cached context length: {ctx:,} tokens for {agent.model}")
        compressor._context_probed = False
        compressor._context_probe_persistable = False


def _accumulate_session_usage(agent, canonical_usage, api_duration):
    agent.session_prompt_tokens += canonical_usage.prompt_tokens
    agent.session_completion_tokens += canonical_usage.output_tokens
    agent.session_total_tokens += canonical_usage.total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens
    # Rolling history for status-bar averages (last 10).
    with suppress(Exception):
        hist = getattr(agent, "_api_latency_history", None)
        if hist is not None:
            hist.append(float(api_duration))
        ohist = getattr(agent, "_api_output_history", None)
        if ohist is not None:
            ohist.append(int(canonical_usage.output_tokens or 0))


def record_response_usage(
    agent: Any, response: Any, *, messages: List[Dict[str, Any]], api_call_count: int,
    api_duration: float, compression_attempts: int, max_compression_attempts: int,
) -> ResponseUsageOutcome:
    """Fold ``response.usage`` into compressor, anchors, session counters, state.db
    and the API-call log line (see module docstring). No-usage responses only
    consume a pending compaction verdict. Returns the loop-visible outcome."""
    compressor = agent.context_compressor
    # Count every completed provider attempt, including providers that omit usage.
    # Token/cost accounting below stays gated on real usage, but the request itself
    # must remain observable.
    agent.session_api_calls += 1
    agent.session_successful_provider_responses += 1
    response_id = getattr(response, "id", None)
    if (isinstance(response_id, str) and not response_id.startswith("stream-")
            and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", response_id)):
        agent.session_provider_request_ids.append(response_id)
    if not (hasattr(response, 'usage') and response.usage):
        from agent.provider_transport_evidence import mark_unknown
        mark_unknown(agent, "provider_response_usage_missing")
        _record_missing_usage(agent, compressor, api_duration)
        return ResponseUsageOutcome(compression_attempts=compression_attempts)

    canonical_usage = normalize_usage(response.usage, provider=agent.provider, api_mode=agent.api_mode)
    # Aggregator-only usage kept for pricing: advisor tokens are priced at each advisor's
    # OWN model rate and added as dollars below.
    aggregator_usage = canonical_usage
    _moa_client, canonical_usage, _moa_ref_cost = _fold_moa_usage(agent, canonical_usage)
    prompt_tokens = canonical_usage.prompt_tokens
    total_tokens = canonical_usage.total_tokens
    # Canonical token + cache buckets for context engines; legacy keys stay for back-compat.
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": canonical_usage.output_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": canonical_usage.reasoning_tokens,
    }
    compression_attempts, rearmed = _update_compressor_usage(
        agent, compressor, messages, aggregator_usage, canonical_usage,
        usage_dict, api_call_count, compression_attempts, max_compression_attempts,
    )

    # Stash canonical usage for on_turn_complete(); keep the latest call's.
    agent._last_turn_usage = dict(usage_dict)
    # The parent's CURRENT prompt size for headroom math (delegate summary budgets): the
    # aggregator's own prompt, never the MoA-folded total (advisor prompts are not in this context).
    agent._last_prompt_size_tokens = int(aggregator_usage.prompt_tokens or 0)

    _persist_confirmed_context(agent, compressor)
    _accumulate_session_usage(agent, canonical_usage, api_duration)
    log_response_usage(agent, response, canonical_usage, api_duration)

    # Price only the aggregator response; MoA advisors retain their own cost provenance.
    cost_delta, cost_status, cost_source = _record_response_cost(
        agent, response, aggregator_usage, _moa_client, _moa_ref_cost,
    )
    from agent.provider_transport_evidence import mark_unknown
    if isinstance(response_id, str) and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", response_id):
        mark_unknown(agent, "provider_runtime_authority_missing")
    else:
        mark_unknown(agent, "provider_response_identity_missing")
    _queue_response_usage(agent, canonical_usage, total_tokens, cost_delta, cost_status, cost_source)

    display_usage(agent, usage_dict, canonical_usage)
    return ResponseUsageOutcome(compression_attempts=compression_attempts, rearmed=rearmed)
