"""Durable execution checkpoint contracts for managed session rollover."""

import importlib
import subprocess

import pytest


def _git(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkpoint_builder():
    try:
        module = importlib.import_module("agent.execution_checkpoint")
    except ModuleNotFoundError:
        pytest.fail("agent.execution_checkpoint is missing")
    return module.build_execution_checkpoint


def _committed_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(
        repo,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=Test",
        "commit",
        "-q",
        "-m",
        "baseline",
    )
    return repo


def test_git_checkpoint_probes_allow_windows_mounted_worktree_latency():
    module = importlib.import_module("agent.execution_checkpoint")

    assert module._GIT_TIMEOUT_SECONDS >= 5.0


def test_build_execution_checkpoint_captures_clean_git_workspace(tmp_path):
    repo = _committed_repo(tmp_path)

    checkpoint = _checkpoint_builder()(cwd=repo, session_id="session-1")

    assert checkpoint["version"] == 1
    assert checkpoint["workspace"] == {
        "cwd": str(repo.resolve()),
        "gitHead": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "branch", "--show-current"),
        "statusSha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    assert checkpoint["patch"] == {
        "kind": "git_diff",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "bytes": 0,
    }
    assert checkpoint["tests"] == {"status": "not_run", "commands": []}
    assert checkpoint["blockers"]["status"] == "clear"
    assert checkpoint["blockers"]["evidence"]
    assert checkpoint["nextAction"]


def test_build_execution_checkpoint_hashes_dirty_workspace(tmp_path):
    repo = _committed_repo(tmp_path)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("preserved\n", encoding="utf-8")

    checkpoint = _checkpoint_builder()(cwd=repo, session_id="session-1")

    assert checkpoint["workspace"]["statusSha256"] != (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert checkpoint["patch"]["bytes"] > 0
    assert checkpoint["patch"]["sha256"] != (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_untracked_file_content_advances_patch_fingerprint(tmp_path):
    repo = _committed_repo(tmp_path)
    untracked = repo / "untracked.txt"
    untracked.write_text("first\n", encoding="utf-8")
    first = _checkpoint_builder()(cwd=repo, session_id="session-1")

    untracked.write_text("second\n", encoding="utf-8")
    second = _checkpoint_builder()(cwd=repo, session_id="session-1")

    assert first["patch"]["bytes"] > 0
    assert first["patch"]["sha256"] != second["patch"]["sha256"]


def test_nested_workspace_checkpoints_the_complete_repository(tmp_path):
    repo = _committed_repo(tmp_path)
    nested = repo / "nested"
    nested.mkdir()
    (repo / "tracked.txt").write_text("changed outside cwd\n", encoding="utf-8")
    untracked = nested / "untracked.txt"
    untracked.write_text("first\n", encoding="utf-8")

    first = _checkpoint_builder()(cwd=nested, session_id="session-1")
    untracked.write_text("second\n", encoding="utf-8")
    second = _checkpoint_builder()(cwd=nested, session_id="session-1")

    assert first is not None
    assert second is not None
    assert first["workspace"]["cwd"] == str(nested.resolve())
    assert first["patch"]["bytes"] > len("nested/untracked.txt") + len("first\n")
    assert first["patch"]["sha256"] != second["patch"]["sha256"]


def test_build_execution_checkpoint_fails_closed_outside_git(tmp_path):
    assert _checkpoint_builder()(cwd=tmp_path, session_id="session-1") is None