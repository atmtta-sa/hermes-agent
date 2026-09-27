"""The per-call log line carries what a cost or cache investigation needs: the cache write count, the
provider's response id, and the serving upstream when the route reports one. Existing parsers read a
prefix of the line, so the fields are appended and optional."""
import logging
from types import SimpleNamespace


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    return AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
                   api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t", platform="cli",
                   quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])


def _usage(read, write, prompt):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=7, total_tokens=prompt + 7,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=read, cache_write_tokens=write),
                           completion_tokens_details=None)


def _line(agent, caplog, resp):
    from agent import turn_usage
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="agent.turn_usage"):
        turn_usage.record_response_usage(agent, resp, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                                         api_duration=0.2, compression_attempts=0, max_compression_attempts=3)
    return next(r.getMessage() for r in caplog.records if r.getMessage().startswith("API call #"))


def test_write_id_and_upstream_are_on_the_line(tmp_path, monkeypatch, caplog):
    a = _agent(tmp_path, monkeypatch)
    try:
        line = _line(a, caplog, SimpleNamespace(usage=_usage(34_283, 28_604, 62_889), id="gen-1788636728-qMa1SbYZcwjrvUzJuF1g",
                                                provider="Anthropic", model="anthropic/claude-fable-5.1"))
    finally:
        a.close()
    assert " cache=34283/62889 (55%)" in line
    assert " write=28604" in line
    assert " id=gen-1788636728-qMa1SbYZcwjrvUzJuF1g" in line
    assert getattr(a, "session_provider_request_ids") == ["gen-1788636728-qMa1SbYZcwjrvUzJuF1g"]
    assert " upstream=Anthropic" in line
    # the pre-existing prefix is unchanged, so older parsers keep matching
    assert line.startswith("API call #1: model=anthropic/claude-fable-5.1 provider=nous in=62889 out=7 total=62896 latency=0.2s")


def test_fields_are_omitted_when_absent(tmp_path, monkeypatch, caplog):
    a = _agent(tmp_path, monkeypatch)
    try:
        line = _line(a, caplog, SimpleNamespace(usage=_usage(0, 0, 100), id=None, model="anthropic/claude-fable-5.1"))
    finally:
        a.close()
    assert "write=" not in line and "id=" not in line and "upstream=" not in line


def test_response_evidence_tracks_missing_usage_separately(tmp_path, monkeypatch, caplog):
    a = _agent(tmp_path, monkeypatch)
    try:
        _line(a, caplog, SimpleNamespace(usage=None, id="response-without-usage"))
        assert a.session_successful_provider_responses == 1
        assert a.session_usage_missing_responses == 1
        assert getattr(a, "session_provider_request_ids") == ["response-without-usage"]
        _line(a, caplog, SimpleNamespace(usage=_usage(0, 0, 100)))
        assert a.session_successful_provider_responses == 2
        assert a.session_usage_missing_responses == 1
    finally:
        a.close()


def test_unknown_cost_remains_incomplete_after_later_priced_response(tmp_path, monkeypatch, caplog):
    a = _agent(tmp_path, monkeypatch)
    from agent import turn_usage

    prices = iter([
        SimpleNamespace(amount_usd=None, status="unknown", source="none"),
        SimpleNamespace(amount_usd=0.01, status="estimated", source="test"),
    ])
    monkeypatch.setattr(turn_usage, "estimate_usage_cost", lambda *_args, **_kwargs: next(prices))
    try:
        _line(a, caplog, SimpleNamespace(usage=_usage(0, 0, 100)))
        _line(a, caplog, SimpleNamespace(usage=_usage(0, 0, 100)))
        assert getattr(a, "session_cost_missing_responses") == 1
        assert a.session_cost_status == "estimated"
        assert a.session_estimated_cost_usd == 0.01
    finally:
        a.close()


def test_estimated_cost_provenance_survives_later_reported_response(tmp_path, monkeypatch, caplog):
    a = _agent(tmp_path, monkeypatch)
    from agent import turn_usage

    prices = iter([
        SimpleNamespace(amount_usd=0.01, status="estimated", source="test"),
        SimpleNamespace(amount_usd=0.02, status="actual", source="provider"),
    ])
    monkeypatch.setattr(turn_usage, "estimate_usage_cost", lambda *_args, **_kwargs: next(prices))
    try:
        _line(a, caplog, SimpleNamespace(usage=_usage(0, 0, 100)))
        _line(a, caplog, SimpleNamespace(usage=_usage(0, 0, 100)))
        assert getattr(a, "session_cost_estimated_responses") == 1
        assert a.session_cost_status == "actual"
    finally:
        a.close()


def test_openrouter_response_charge_overrides_estimate_only_on_verified_route(tmp_path, monkeypatch, caplog):
    from agent import turn_usage
    monkeypatch.setattr(
        turn_usage, "estimate_usage_cost",
        lambda *_args, **_kwargs: SimpleNamespace(amount_usd=0.01, status="estimated", source="test"),
    )
    for base_url, expected_cost, expected_status in (
        ("https://openrouter.ai/api/v1", 0.00017, "actual"),
        ("https://openrouter.ai.evil.example/api/v1", 0.01, "estimated"),
    ):
        agent = _agent(tmp_path, monkeypatch)
        agent.base_url = base_url
        usage = _usage(0, 0, 100)
        usage.cost = 0.00017
        try:
            _line(agent, caplog, SimpleNamespace(usage=usage, id="gen-1"))
            assert agent.session_estimated_cost_usd == expected_cost
            assert agent.session_cost_status == expected_status
            assert agent.session_cost_estimated_responses == (expected_status != "actual")
        finally:
            agent.close()


def test_openrouter_aggregator_charge_cannot_certify_estimated_advisor_spend(tmp_path, monkeypatch, caplog):
    from agent import turn_usage
    agent = _agent(tmp_path, monkeypatch)
    agent.base_url = "https://openrouter.ai/api/v1"
    monkeypatch.setattr(
        turn_usage, "_fold_moa_usage",
        lambda _agent, usage: (None, usage, 0.004),
    )
    usage = _usage(0, 0, 100)
    usage.cost = 0.00017
    try:
        _line(agent, caplog, SimpleNamespace(usage=usage, id="gen-1"))
        assert agent.session_cost_estimated_responses == 1
    finally:
        agent.close()


def test_synthetic_stream_id_is_not_provider_request_evidence(tmp_path, monkeypatch, caplog):
    agent = _agent(tmp_path, monkeypatch)
    try:
        _line(agent, caplog, SimpleNamespace(usage=_usage(0, 0, 100), id="stream-1234"))
        assert agent.session_provider_request_ids == []
    finally:
        agent.close()


def test_forensics_parser_reads_the_new_fields(tmp_path):
    from evals.postmortem.forensics.logcalls import parse_logs
    log = tmp_path / "agent.log"
    log.write_text(
        "2026-09-06 19:32:08,549 INFO [s1] agent.conversation_loop: API call #3: model=anthropic/claude-fable-5.1 provider=nous "
        "in=62889 out=7 total=62896 latency=0.2s cache=34283/62889 (55%) write=28604 id=gen-1788636728-qMa1 upstream=Claude Platform on AWS\n"
        "2026-09-06 19:32:09,549 INFO [s1] agent.conversation_loop: API call #4: model=anthropic/claude-fable-5.1 provider=nous "
        "in=91079 out=7 total=91086 latency=0.2s cache=62887/91079 (69%)\n", encoding="utf-8")
    calls = parse_logs([str(log)], {"s1"})
    assert [c["n"] for c in calls] == [3, 4]
    assert calls[0]["write"] == 28604 and calls[0]["id"] == "gen-1788636728-qMa1" and calls[0]["upstream"] == "Claude Platform on AWS"
    assert "write" not in calls[1] and "id" not in calls[1]
