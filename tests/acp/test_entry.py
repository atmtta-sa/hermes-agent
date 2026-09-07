"""Tests for acp_adapter.entry startup wiring."""

import logging
import sys

import acp
import pytest

from acp_adapter import entry


class _FilePipelineHandler(logging.Handler):
    """Stand-in for Hermes's queued agent.log pipeline."""


def test_setup_logging_preserves_file_pipeline_and_replaces_stream_handler(monkeypatch):
    root = logging.getLogger()
    prior_handlers = root.handlers[:]
    prior_level = root.level
    file_pipeline = _FilePipelineHandler()
    old_stream = logging.StreamHandler()
    monkeypatch.setattr(entry.sys, "stderr", object())
    root.handlers = [file_pipeline, old_stream]

    try:
        entry._setup_logging()

        assert file_pipeline in root.handlers
        assert old_stream not in root.handlers
        streams = [handler for handler in root.handlers if isinstance(handler, logging.StreamHandler)]
        assert len(streams) == 1
        assert streams[0].stream is entry.sys.stderr
    finally:
        root.handlers = prior_handlers
        root.setLevel(prior_level)


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {}

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_main_skips_configured_mcp_discovery_when_requested(monkeypatch):
    discovery_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "tools.mcp_tool_discovery.discover_mcp_tools",
        lambda: discovery_calls.append(True),
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert discovery_calls == []










def test_main_setup_offers_browser_install_when_tty(monkeypatch):
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    bootstrap_calls = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: bootstrap_calls.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert bootstrap_calls == [False]










def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    def fake_ensure(dep, interactive=True):
        return dep != "browser"  # browser fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1
