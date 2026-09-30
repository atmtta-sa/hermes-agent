"""Provider-free interception of final physical LLM request kwargs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
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
    sink = vars(agent).get("_effective_request_capture_sink")
    if not callable(sink):
        return
    sink({
        "transport": transport,
        "provider": str(getattr(agent, "provider", "") or ""),
        "model": str(request.get("model") or getattr(agent, "model", "") or ""),
        "request": request,
    })
    raise EffectiveRequestCaptured("Final provider request captured; transport skipped")
