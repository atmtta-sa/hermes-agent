"""Provider-free interception of final physical LLM request kwargs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from typing import Any


_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "proxy_authorization",
    "set_cookie",
}


class EffectiveRequestCaptured(RuntimeError):
    """The final request was captured intentionally instead of being transmitted."""


class EffectiveRequestPolicyDenied(RuntimeError):
    """The final request violated a required pretransport context policy."""


def _jsonable_object(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json"))
    public_state = (
        {
            str(key): item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
        if hasattr(value, "__dict__")
        else {}
    )
    if public_state:
        return {"type": type(value).__name__, "value": _jsonable(public_state)}
    return {"type": type(value).__name__, "value": str(value)}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return {"type": type(value).__name__, "hex": bytes(value).hex()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return _jsonable_object(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _redact(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            redacted[key] = (
                "[REDACTED]"
                if normalized in _SENSITIVE_KEYS or normalized.endswith("_api_key")
                else _redact(item)
            )
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(value, force=True)
    return value


def _model_visible_sections(request: Any) -> tuple[Any, Any]:
    if not isinstance(request, dict):
        return [], []
    extra_body = request.get("extra_body") if isinstance(request, dict) else None
    extra_body = extra_body if isinstance(extra_body, dict) else {}
    messages = (
        request.get("messages") or request.get("input") or extra_body.get("input") or []
    )
    tools = request.get("tools") or extra_body.get("tools") or []
    return messages, tools


def _manifest_request_lists(request: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    extra_body = request.get("extra_body")
    extra_body = extra_body if isinstance(extra_body, dict) else {}
    messages = request.get("messages") or request.get("input") or extra_body.get("input") or []
    tools = request.get("tools") or extra_body.get("tools") or []
    return (
        messages if isinstance(messages, list) else [],
        tools if isinstance(tools, list) else [],
    )


def _split_system_messages(messages: list[Any]) -> tuple[list[Any], list[Any]]:
    system = []
    non_system = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            system.append(message.get("content"))
        else:
            non_system.append(message)
    return system, non_system


def _manifest_sections(request: Any) -> tuple[Any, Any, Any, int, int]:
    if not isinstance(request, dict):
        return [], [], [], 0, 0
    messages, tools = _manifest_request_lists(request)
    system_messages, non_system_messages = _split_system_messages(messages)
    instructions = request.get("instructions")
    system = instructions if isinstance(instructions, str) else system_messages
    return system, non_system_messages, tools, len(messages), len(tools)


def _text_blocks(value: Any):
    if isinstance(value, str):
        yield value.replace("\r\n", "\n").replace("\r", "\n")
    elif isinstance(value, dict):
        for item in value.values():
            yield from _text_blocks(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _text_blocks(item)


def _duplicate_blocks(*sections: Any, minimum_bytes: int = 512) -> list[dict[str, Any]]:
    counts: dict[str, tuple[int, int]] = {}
    for section in sections:
        for block in _text_blocks(section):
            encoded = block.encode("utf-8")
            if len(encoded) < minimum_bytes:
                continue
            digest = hashlib.sha256(encoded).hexdigest()
            size, occurrences = counts.get(digest, (len(encoded), 0))
            counts[digest] = (size, occurrences + 1)
    return [
        {"sha256": digest, "bytes": size, "occurrences": occurrences}
        for digest, (size, occurrences) in sorted(counts.items())
        if occurrences > 1
    ]


def build_effective_request_manifest(
    capture: dict[str, Any], *, minimum_duplicate_block_bytes: int = 512
) -> dict[str, Any]:
    """Return deterministic measurements and hashes without request content."""
    request = capture["request"]
    system, messages, tools, message_count, tool_count = _manifest_sections(request)
    request_bytes = _canonical_bytes(request)
    system_bytes = _canonical_bytes(system)
    messages_bytes = _canonical_bytes(messages)
    tools_bytes = _canonical_bytes(tools)
    return {
        "schema": "hermes.effective_request_manifest.v1",
        "transport": str(capture.get("transport") or ""),
        "provider": str(capture.get("provider") or ""),
        "model": str(capture.get("model") or ""),
        "measurements": {
            "request_bytes": len(request_bytes),
            "system_bytes": len(system_bytes),
            "messages_bytes": len(messages_bytes),
            "tools_bytes": len(tools_bytes),
            "message_count": message_count,
            "tool_count": tool_count,
        },
        "hashes": {
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "system_sha256": hashlib.sha256(system_bytes).hexdigest(),
            "messages_sha256": hashlib.sha256(messages_bytes).hexdigest(),
            "tools_sha256": hashlib.sha256(tools_bytes).hexdigest(),
        },
        "duplicate_blocks": _duplicate_blocks(
            system, messages, minimum_bytes=minimum_duplicate_block_bytes
        ),
    }


_POLICY_FIELDS = {
    "required",
    "maxRequestBytes",
    "maxSystemBytes",
    "maxMessagesBytes",
    "maxToolsBytes",
    "maxToolCount",
    "maxExactDuplicateBlocks",
    "minDuplicateBlockBytes",
}


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validated_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _POLICY_FIELDS:
        raise EffectiveRequestPolicyDenied("effective_request_policy_invalid")
    if not isinstance(value["required"], bool):
        raise EffectiveRequestPolicyDenied("effective_request_policy_invalid")
    if not all(
        _positive_integer(value[key])
        for key in (
            "maxRequestBytes",
            "maxSystemBytes",
            "maxMessagesBytes",
            "maxToolsBytes",
            "minDuplicateBlockBytes",
        )
    ):
        raise EffectiveRequestPolicyDenied("effective_request_policy_invalid")
    if not all(
        _nonnegative_integer(value[key])
        for key in ("maxToolCount", "maxExactDuplicateBlocks")
    ):
        raise EffectiveRequestPolicyDenied("effective_request_policy_invalid")
    return value


def _effective_request_policy(agent: Any) -> dict[str, Any] | None:
    configured = vars(agent).get("_effective_request_policy")
    if configured is not None:
        return _validated_policy(configured)
    raw = os.environ.get("HERMES_EFFECTIVE_REQUEST_POLICY_JSON", "").strip()
    if not raw:
        return None
    try:
        configured = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EffectiveRequestPolicyDenied("effective_request_policy_invalid") from exc
    policy = _validated_policy(configured)
    agent._effective_request_policy = policy
    return policy


def _policy_denial_reason(
    manifest: dict[str, Any], policy: dict[str, Any]
) -> str | None:
    measurements = manifest["measurements"]
    limits = (
        ("request_bytes", "maxRequestBytes", "effective_request_request_bytes_exceeded"),
        ("system_bytes", "maxSystemBytes", "effective_request_system_bytes_exceeded"),
        ("messages_bytes", "maxMessagesBytes", "effective_request_messages_bytes_exceeded"),
        ("tools_bytes", "maxToolsBytes", "effective_request_tools_bytes_exceeded"),
        ("tool_count", "maxToolCount", "effective_request_tool_count_exceeded"),
    )
    for measurement, limit, reason in limits:
        if measurements[measurement] > policy[limit]:
            return reason
    if len(manifest["duplicate_blocks"]) > policy["maxExactDuplicateBlocks"]:
        return "effective_request_duplicate_blocks_exceeded"
    return None


def _enforce_effective_request_policy(
    agent: Any, capture: dict[str, Any]
) -> None:
    policy = _effective_request_policy(agent)
    if policy is None:
        return
    manifest = build_effective_request_manifest(
        capture,
        minimum_duplicate_block_bytes=policy["minDuplicateBlockBytes"],
    )
    reason = _policy_denial_reason(manifest, policy)
    manifest["policy_sha256"] = hashlib.sha256(_canonical_bytes(policy)).hexdigest()
    manifest["verdict"] = "denied" if reason else "allowed"
    manifest["reason"] = reason
    agent._effective_request_manifest = manifest
    manifests = vars(agent).setdefault("_effective_request_manifests", [])
    manifests.append(manifest)
    if reason:
        from agent.provider_transport_evidence import deny_pretransport

        deny_pretransport(agent, reason)
        raise EffectiveRequestPolicyDenied(reason)


def build_run_context_telemetry(agent: Any) -> dict[str, Any] | None:
    """Return all measured requests for this process without request content."""
    manifests = list(vars(agent).get("_effective_request_manifests") or [])
    if not manifests:
        return None
    denied = next(
        (manifest for manifest in manifests if manifest.get("verdict") == "denied"),
        None,
    )
    return {
        "schema": "hermes.context_telemetry.v1",
        "request_count": len(manifests),
        "verdict": "denied" if denied else "allowed",
        "reason": denied.get("reason") if denied else None,
        "requests": manifests,
    }


def build_effective_request_artifact(capture: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic redacted artifact derived from exact captured kwargs."""
    request = capture["request"]
    request_bytes = _canonical_bytes(request)
    messages, tools = _model_visible_sections(request)
    return {
        "schema": "hermes.effective_request.v1",
        "transport": str(capture.get("transport") or ""),
        "provider": str(capture.get("provider") or ""),
        "model": str(capture.get("model") or ""),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "measurements": {
            "request_bytes": len(request_bytes),
            "messages_bytes": len(_canonical_bytes(messages)),
            "tools_bytes": len(_canonical_bytes(tools)),
        },
        "request": _redact(request),
    }


def render_effective_request(
    agent: Any,
    user_message: Any,
    *,
    system_message: str | None = None,
) -> dict[str, Any]:
    """Run normal first-turn assembly and return its final request without transport."""
    if getattr(agent, "api_mode", None) not in {"chat_completions", "codex_responses"}:
        raise ValueError(
            "Effective-request rendering is not implemented for this API mode"
        )
    captured: list[dict[str, Any]] = []
    missing = object()
    previous = getattr(agent, "_effective_request_capture_sink", missing)
    agent._effective_request_capture_sink = captured.append
    try:
        agent.run_conversation(user_message, system_message=system_message)
    except EffectiveRequestCaptured:
        if len(captured) != 1:
            raise RuntimeError(
                "Effective-request rendering did not capture exactly one request"
            )
        return build_effective_request_artifact(captured[0])
    finally:
        if previous is missing:
            delattr(agent, "_effective_request_capture_sink")
        else:
            agent._effective_request_capture_sink = previous
    raise RuntimeError("Provider path completed without effective-request capture")


def capture_final_provider_request(
    agent: Any,
    request: dict[str, Any],
    *,
    transport: str,
) -> None:
    """Deliver exact final kwargs to an explicit sink, then stop before transport."""
    capture = {
        "transport": transport,
        "provider": str(getattr(agent, "provider", "") or ""),
        "model": str(request.get("model") or getattr(agent, "model", "") or ""),
        "request": request,
    }
    _enforce_effective_request_policy(agent, capture)
    sink = vars(agent).get("_effective_request_capture_sink")
    if callable(sink):
        sink(capture)
        raise EffectiveRequestCaptured("Final provider request captured; transport skipped")
    from agent.provider_transport_evidence import dispatch_transport

    dispatch_transport(agent)
