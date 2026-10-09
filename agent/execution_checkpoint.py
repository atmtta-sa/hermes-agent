"""Mechanically verified workspace checkpoint for managed session rollover."""

from __future__ import annotations

import hashlib
import logging
import os
import stat
from pathlib import Path
from typing import Any, Sequence

from hermes_cli._subprocess_compat import bounded_probe_run, noninteractive_git_env


logger = logging.getLogger(__name__)
_GIT_TIMEOUT_SECONDS = 5.0
_MAX_UNTRACKED_BYTES = 67_108_864


def _git_bytes(cwd: Path, args: Sequence[str]) -> bytes | None:
    result = bounded_probe_run(
        ["git", "-C", str(cwd), *args],
        timeout=_GIT_TIMEOUT_SECONDS,
        errors="surrogateescape",
        env=noninteractive_git_env(),
    )
    if result is None or result.returncode != 0:
        return None
    return (result.stdout or "").encode("utf-8", "surrogateescape")


def _git_text(cwd: Path, args: Sequence[str]) -> str | None:
    raw = _git_bytes(cwd, args)
    return None if raw is None else raw.decode("utf-8", "surrogateescape").strip()


def _verification_tests(cwd: Path, session_id: str | None) -> dict[str, Any]:
    try:
        from agent.verification_evidence import verification_status

        status = verification_status(session_id=session_id, cwd=cwd)
    except Exception:
        logger.debug("rollover verification evidence unavailable", exc_info=True)
        return {"status": "not_run", "commands": []}

    evidence = status.get("evidence") if isinstance(status, dict) else None
    state = status.get("status") if isinstance(status, dict) else None
    if state not in {"passed", "failed"} or not isinstance(evidence, dict):
        return {"status": "not_run", "commands": []}
    command = evidence.get("canonical_command")
    exit_code = evidence.get("exit_code")
    if not isinstance(command, str) or not command or not isinstance(exit_code, int):
        return {"status": "not_run", "commands": []}
    return {
        "status": state,
        "commands": [{"command": command, "exitCode": exit_code}],
    }


def _hash_regular_file(digest: Any, path: Path, metadata: os.stat_result) -> int | None:
    if metadata.st_size > _MAX_UNTRACKED_BYTES:
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    measured_bytes = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != metadata.st_size:
            return None
        while chunk := os.read(descriptor, 1_048_576):
            digest.update(chunk)
            measured_bytes += len(chunk)
        closed = os.fstat(descriptor)
        if (closed.st_size, closed.st_mtime_ns) != (metadata.st_size, metadata.st_mtime_ns):
            return None
        return measured_bytes
    finally:
        os.close(descriptor)


def _hash_untracked_path(digest: Any, root: Path, raw_path: bytes) -> int | None:
    path = root / os.fsdecode(raw_path)
    metadata = path.lstat()
    digest.update(b"\0hermes-untracked\0")
    digest.update(len(raw_path).to_bytes(8, "big"))
    digest.update(raw_path)
    if stat.S_ISLNK(metadata.st_mode):
        content = os.fsencode(os.readlink(path))
        digest.update(b"symlink\0")
        digest.update(content)
        return len(raw_path) + len(content)
    if not stat.S_ISREG(metadata.st_mode):
        return None
    digest.update(b"file\0")
    content_bytes = _hash_regular_file(digest, path, metadata)
    return None if content_bytes is None else len(raw_path) + content_bytes


def _patch_evidence(cwd: Path, root: Path, tracked_patch: bytes) -> dict[str, Any] | None:
    untracked = _git_bytes(cwd, ["ls-files", "-z", "--others", "--exclude-standard"])
    if untracked is None:
        return None
    digest = hashlib.sha256(tracked_patch)
    measured_bytes = len(tracked_patch)
    for raw_path in sorted(path for path in untracked.split(b"\0") if path):
        path_bytes = _hash_untracked_path(digest, root, raw_path)
        if path_bytes is None:
            return None
        measured_bytes += path_bytes
    return {"kind": "git_diff", "sha256": digest.hexdigest(), "bytes": measured_bytes}


def build_execution_checkpoint(
    *, cwd: str | Path, session_id: str | None
) -> dict[str, Any] | None:
    """Return a version-1 rollover checkpoint, or ``None`` when Git state is unprovable."""
    try:
        workspace = Path(cwd).expanduser().resolve()
        root = _git_text(workspace, ["rev-parse", "--show-toplevel"])
        if not root:
            return None
        repository = Path(root)
        head = _git_text(repository, ["rev-parse", "HEAD"])
        branch = _git_text(repository, ["branch", "--show-current"])
        status = _git_bytes(
            repository,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        )
        patch = _git_bytes(
            repository,
            ["diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--"],
        )
        if not head or branch is None or status is None or patch is None:
            return None
        patch_evidence = _patch_evidence(repository, repository, patch)
        if patch_evidence is None:
            return None
        return {
            "version": 1,
            "workspace": {
                "cwd": str(workspace),
                "gitHead": head,
                "branch": branch or None,
                "statusSha256": hashlib.sha256(status).hexdigest(),
            },
            "patch": patch_evidence,
            "tests": _verification_tests(Path(root), session_id),
            "blockers": {
                "status": "clear",
                "evidence": [
                    "The runtime stopped only to cross the managed session boundary."
                ],
            },
            "nextAction": (
                "Continue the current issue from this durable workspace without repeating "
                "completed external side effects."
            ),
        }
    except Exception:
        logger.debug("could not build durable rollover checkpoint", exc_info=True)
        return None