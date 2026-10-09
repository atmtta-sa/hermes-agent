"""Paid one-shot runs must distinguish process completion from useful work."""

import json

from hermes_cli.oneshot import _write_usage_file


def test_exit_zero_without_provider_work_is_no_progress(tmp_path):
    path = tmp_path / "usage.json"

    _write_usage_file(
        str(path),
        {
            "provider": "openrouter",
            "model": "moonshotai/kimi-k3",
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "completed": True,
            "failed": False,
        },
    )

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["process_status"] == "succeeded"
    assert report["work_outcome"] == "no_progress"
    assert report["successful_provider_responses"] == 0


def test_paid_provider_response_without_usage_is_telemetry_missing(tmp_path):
    path = tmp_path / "usage.json"

    _write_usage_file(
        str(path),
        {
            "provider": "openrouter",
            "model": "moonshotai/kimi-k3",
            "api_calls": 1,
            "input_tokens": None,
            "output_tokens": None,
            "completed": True,
            "failed": False,
        },
    )

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["work_outcome"] == "telemetry_missing"
    assert report["telemetry_complete"] is False