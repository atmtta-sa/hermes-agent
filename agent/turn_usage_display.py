"""Per-response usage diagnostics; never decide accounting or billing here."""

import logging
from contextlib import suppress

logger = logging.getLogger("agent.conversation_loop")


def usage_log_suffix(response, canonical_usage):
    prompt_tokens = canonical_usage.prompt_tokens
    cache_pct = ""
    if canonical_usage.cache_read_tokens and prompt_tokens:
        cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
    if canonical_usage.cache_write_tokens:
        cache_pct += f" write={canonical_usage.cache_write_tokens}"
    rid = getattr(response, "id", None)
    ident = f" id={rid}" if isinstance(rid, str) and rid else ""
    upstream = getattr(response, "provider", None)
    if isinstance(upstream, str) and upstream:
        ident += f" upstream={upstream}"
    return cache_pct, ident


def maybe_switch_nous_wire(agent, response):
    # nous.anthropic_wire=auto: the session's wire is decided once, from this first response.
    if agent.session_api_calls == 1 and (agent.provider or "") == "nous":
        with suppress(Exception):
            from agent.nous_wire import maybe_switch_wire_after_first_response
            maybe_switch_wire_after_first_response(agent, response, agent.session_api_calls)


def log_response_usage(agent, response, canonical_usage, api_duration):
    cache_pct, ident = usage_log_suffix(response, canonical_usage)
    logger.info(
        "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s%s",
        agent.session_api_calls, agent.model, agent.provider or "unknown",
        canonical_usage.prompt_tokens, canonical_usage.output_tokens, canonical_usage.total_tokens,
        api_duration, cache_pct, ident,
    )
    maybe_switch_nous_wire(agent, response)


def display_usage(agent, usage_dict, canonical_usage):
    if agent.verbose_logging:
        logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
    # Report cache stats for any provider returning prompt_tokens_details.cached_tokens.
    cached = canonical_usage.cache_read_tokens
    written = canonical_usage.cache_write_tokens
    prompt = usage_dict["prompt_tokens"]
    if (cached or written) and not agent.quiet_mode:
        hit_pct = (cached / prompt * 100) if prompt > 0 else 0
        agent._vprint(
            f"{agent.log_prefix}   💾 Cache: "
            f"{cached:,}/{prompt:,} tokens "
            f"({hit_pct:.0f}% hit, {written:,} written)"
        )
