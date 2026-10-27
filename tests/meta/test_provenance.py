"""Tests for text2sql.meta.provenance."""

import hashlib
import platform
import subprocess
import types
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from text2sql.meta import provenance

START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs in a temporary directory, so that the git tests never touch this repository."""
    monkeypatch.chdir(tmp_path)


def test_attempt_returns_the_result_or_none() -> None:
    assert provenance.attempt(int, "5") == 5

    assert provenance.attempt(int, "x") is None


def testcached_revision_is_the_snapshot_folder(
    stub_module: Callable[..., types.ModuleType],
) -> None:
    stub_module(
        "huggingface_hub", try_to_load_from_cache=lambda *_: "/c/snapshots/abc123/config.json"
    )

    assert provenance._cached_revision("some/model") == "abc123"


def test_cached_revision_raises_when_the_file_is_not_cached(
    stub_module: Callable[..., types.ModuleType],
) -> None:
    stub_module("huggingface_hub", try_to_load_from_cache=lambda *_: None)

    with pytest.raises(FileNotFoundError):
        provenance._cached_revision("some/model")


def test_environment_package_versions_are_null_for_missing_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def version(name: str) -> str:
        if name == "torch":
            return "2.0"
        raise provenance.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(provenance.metadata, "version", version)

    versions = provenance.environment(("torch", "transformers"))["packages"]

    assert versions["torch"] == "2.0"
    assert versions["transformers"] is None


def test_git_commit_is_the_head_of_the_repository(tmp_path: Path) -> None:
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()

    assert provenance.git_commit() == head


@pytest.mark.parametrize(
    ("url", "repo"),
    [
        (
            "https://github.com/unbiasederudite/qwen2.5-coder-text2sql-qlora.git",
            "unbiasederudite/qwen2.5-coder-text2sql-qlora",
        ),
        ("git@github.com:owner/name.git", "owner/name"),
        ("https://github.com/owner/name", "owner/name"),
        (None, None),
        ("https://example.com/owner/name.git", None),
    ],
)
def test_repository_is_read_from_the_origin_remote(
    url: str | None, repo: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance, "_git", lambda *args: url)

    assert provenance.repository() == repo


def test_git_commit_is_none_outside_a_repository(tmp_path: Path) -> None:
    assert provenance.git_commit() is None  # tmp_path is no repository


def test_git_is_none_when_git_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> str:
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "check_output", missing)

    assert provenance._git("rev-parse", "HEAD") is None


def test_environment_lists_the_requested_packages_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance.metadata, "version", lambda name: "1.0")

    packages = provenance.environment(("peft", "trl"))["packages"]

    assert packages == {"peft": "1.0", "trl": "1.0"}


def test_environment_describes_the_code_and_the_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "_git", lambda *args: None)  # no repository

    info = provenance.environment(("peft",))

    assert list(info) == ["repo", "commit", "python", "packages", "gpu"]
    assert info["repo"] is None and info["commit"] is None
    assert info["python"] == platform.python_version()
    assert list(info["packages"]) == ["peft"]  # type: ignore[call-overload]


def test_timing_gives_the_start_the_end_and_the_duration() -> None:
    assert provenance.timing(START, START + timedelta(seconds=90)) == {
        "started_at": "2026-01-01T12:00:00+00:00",
        "finished_at": "2026-01-01T12:01:30+00:00",
        "duration_seconds": 90,
    }


def test_sha256_hashes_the_file_content(tmp_path: Path) -> None:
    file = tmp_path / "f"
    file.write_bytes(b"abc")

    assert provenance.sha256(file) == hashlib.sha256(b"abc").hexdigest()
