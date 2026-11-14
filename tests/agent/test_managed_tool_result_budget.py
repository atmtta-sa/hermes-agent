import json
from types import SimpleNamespace

from agent import tool_executor
from tools import tool_result_storage
from tools.budget_config import managed_tool_result_budget_from_env


def test_managed_budget_env_overrides_context_defaults(monkeypatch):
    monkeypatch.setenv(
        "HERMES_TOOL_RESULT_BUDGET_JSON",
        json.dumps(
            {
                "perResultBytes": 8_000,
                "perTurnBytes": 24_000,
                "perSessionBytes": 48_000,
            }
        ),
    )
    agent = SimpleNamespace(
        context_compressor=SimpleNamespace(context_length=200_000),
    )

    config = tool_executor._budget_for_agent(agent)
    managed = managed_tool_result_budget_from_env()

    assert config.default_result_size == 8_000
    assert config.turn_budget == 24_000
    assert managed is not None
    assert managed.per_session_bytes == 48_000


def test_byte_budget_spills_largest_inline_result(monkeypatch, tmp_path):
    def fake_spill(content, filename):
        path = tmp_path / filename
        path.write_text(content)
        return str(path)

    monkeypatch.setattr(tool_result_storage, "_write_to_spillover", fake_spill)
    messages = [
        {"role": "tool", "tool_call_id": "one", "content": "é" * 4_000},
        {"role": "tool", "tool_call_id": "two", "content": "x" * 6_000},
    ]

    tool_result_storage.enforce_byte_budget(messages, max_bytes=10_000)

    assert sum(len(message["content"].encode("utf-8")) for message in messages) <= 10_000
    assert any("<persisted-output>" in message["content"] for message in messages)


def test_byte_budget_truncates_read_file_without_spill():
    messages = [
        {
            "role": "tool",
            "tool_name": "read_file",
            "tool_call_id": "read-one",
            "content": "é" * 6_000,
        }
    ]

    tool_result_storage.enforce_byte_budget(messages, max_bytes=8_000)

    assert len(messages[0]["content"].encode("utf-8")) <= 8_000
    assert "<persisted-output>" not in messages[0]["content"]
    assert "managed tool-result byte budget" in messages[0]["content"]


def test_invalid_managed_budget_fails_closed(monkeypatch):
    monkeypatch.setenv(
        "HERMES_TOOL_RESULT_BUDGET_JSON",
        json.dumps(
            {
                "perResultBytes": 24_000,
                "perTurnBytes": 8_000,
                "perSessionBytes": 48_000,
            }
        ),
    )

    assert managed_tool_result_budget_from_env() is None
