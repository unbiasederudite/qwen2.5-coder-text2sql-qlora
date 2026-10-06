"""Tests for text2sql.meta."""

import hashlib
import json
import sqlite3
import subprocess
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from text2sql import meta

START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
GENERATION = {"max_new_tokens": 256, "do_sample": False}


@pytest.fixture(autouse=True)
def fake_model_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stands in for `text2sql.model`, which needs torch and transformers."""
    module = types.ModuleType("text2sql.model")
    module.__dict__["GENERATION"] = GENERATION
    monkeypatch.setitem(sys.modules, "text2sql.model", module)


def fake_model(quantization: object = None) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(name_or_path="some/model", quantization_config=quantization)
    )


def write(
    tmp_path: Path,
    model: object | None = None,
    finished_at: datetime = START + timedelta(seconds=90),
) -> dict[str, Any]:
    predictions = tmp_path / "predictions" / "m_run_dev.jsonl"
    predictions.parent.mkdir()
    predictions.write_text('{"id": 0}\n')
    model = model or fake_model()
    path = meta.write_generation_metadata(
        predictions,
        model,
        "run",
        "dev",
        16,
        5,
        2,
        START,
        finished_at,
    )
    assert path == predictions.with_suffix(".meta.json")
    result: dict[str, Any] = json.loads(path.read_text())
    return result


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def test_attempt_returns_the_result_or_none() -> None:
    assert meta.attempt(int, "5") == 5

    assert meta.attempt(int, "x") is None


def test_sha256_hashes_the_file_content(tmp_path: Path) -> None:
    file = tmp_path / "f"
    file.write_bytes(b"abc")

    assert meta.sha256(file) == hashlib.sha256(b"abc").hexdigest()


def test_write_generation_metadata_records_the_run(tmp_path: Path) -> None:
    info = write(tmp_path)

    assert info["model"] == "some/model"
    assert (info["run"], info["split"]) == ("run", "dev")
    assert (info["questions"], info["already_done"]) == (5, 2)
    assert info["generation"] == {**GENERATION, "batch_size": 16}
    assert info["started_at"] == "2026-01-01T12:00:00+00:00"
    assert info["finished_at"] == "2026-01-01T12:01:30+00:00"
    assert info["duration_seconds"] == 90
    assert info["predictions_sha256"] == hashlib.sha256(b'{"id": 0}\n').hexdigest()


def test_write_generation_metadata_leaves_unreadable_fields_null(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    info = write(tmp_path)

    assert info["split_file_sha256"] is None  # no data/spider_data/dev.json here
    assert info["model_revision"] is None  # nothing is cached for the model
    assert "split_file_sha256" in capsys.readouterr().out


def test_write_generation_metadata_records_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(meta, "git", lambda *args: "https://github.com/owner/name.git")

    assert write(tmp_path)["repo"] == "owner/name"


def test_write_generation_metadata_serializes_the_quantization_config(tmp_path: Path) -> None:
    class Config:
        def to_dict(self) -> dict[str, object]:
            return {"load_in_4bit": True}

    model = fake_model(Config())

    assert write(tmp_path, model)["quantization"] == {"load_in_4bit": True}


def test_cached_revision_is_the_snapshot_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = types.ModuleType("huggingface_hub")
    hub.__dict__["try_to_load_from_cache"] = lambda *_: "/c/snapshots/abc123/config.json"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert meta.cached_revision("some/model") == "abc123"


def test_package_versions_are_null_for_missing_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    def version(name: str) -> str:
        if name == "torch":
            return "2.0"
        raise meta.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(meta.metadata, "version", version)

    versions = meta.package_versions()

    assert versions["torch"] == "2.0"
    assert versions["transformers"] is None


def test_git_commit_is_the_head_of_the_repository(tmp_path: Path) -> None:
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()

    assert meta.git_commit() == head


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
    monkeypatch.setattr(meta, "git", lambda *args: url)

    assert meta.repository() == repo


def test_git_commit_is_none_outside_a_repository(tmp_path: Path) -> None:
    assert meta.git_commit() is None  # tmp_path is no repository


def predictions_file(tmp_path: Path, info: dict[str, object] | None) -> Path:
    """Writes `run.jsonl` and, if given, `run.meta.json` with the hash of the predictions."""
    path = tmp_path / "run.jsonl"
    path.write_text('{"id": 0}\n')
    if info is not None:
        (tmp_path / "run.meta.json").write_text(
            json.dumps({"predictions_sha256": meta.sha256(path), **info})
        )
    return path


def test_read_generation_metadata_reads_the_file_next_to_the_predictions(tmp_path: Path) -> None:
    assert meta.read_generation_metadata(predictions_file(tmp_path, None), "dev") is None

    info = meta.read_generation_metadata(predictions_file(tmp_path, {"run": "r"}), "dev")

    assert info is not None and info["run"] == "r"


def test_read_generation_metadata_stops_when_it_describes_another_predictions_file(
    tmp_path: Path,
) -> None:
    path = predictions_file(tmp_path, {})
    (tmp_path / "run.meta.json").write_text(json.dumps({"predictions_sha256": "0" * 64}))

    with pytest.raises(ValueError, match="describes another predictions file"):
        meta.read_generation_metadata(path, "dev")


def test_read_generation_metadata_accepts_metadata_without_a_hash(tmp_path: Path) -> None:
    path = predictions_file(tmp_path, {})
    (tmp_path / "run.meta.json").write_text(json.dumps({"run": "r"}))

    assert meta.read_generation_metadata(path, "dev") == {"run": "r"}


def test_read_generation_metadata_warns_about_another_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = predictions_file(tmp_path, {"split": "test"})

    meta.read_generation_metadata(path, "dev")

    assert "metadata is for the test split, scoring dev" in capsys.readouterr().err


def test_write_scoring_metadata_records_the_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = {
        ("remote", "get-url", "origin"): "git@github.com:owner/name.git",
        ("rev-parse", "HEAD"): "abc123",
    }
    monkeypatch.setattr(meta, "git", lambda *args: answers.get(args))
    given = {
        "model": "m",
        "model_revision": "r1",
        "run": "baseline",
        "generation": {"batch_size": 16},
    }
    predictions = predictions_file(tmp_path, None)

    path = meta.write_scoring_metadata(
        tmp_path / "x.report.json", predictions, "dev", True, False, given
    )

    assert path == tmp_path / "x.report.meta.json"
    assert json.loads(path.read_text()) == {
        "split": "dev",
        "predictions_sha256": meta.sha256(predictions),
        "model": "m",
        "model_revision": "r1",
        "run": "baseline",  # the generation settings stay in the metadata of the predictions
        "keep_distinct": True,
        "extraction": False,
        "sqlite_version": sqlite3.sqlite_version,
        "repo": "owner/name",
        "commit": "abc123",
    }


def test_write_scoring_metadata_without_generation_metadata_or_git_has_nulls(
    tmp_path: Path,
) -> None:
    predictions = predictions_file(tmp_path, None)

    path = meta.write_scoring_metadata(
        tmp_path / "x.report.json", predictions, "dev", False, True, None
    )

    info = json.loads(path.read_text())  # tmp_path is no repository
    nulls = ("model", "model_revision", "run", "repo", "commit")
    assert [info[name] for name in nulls] == [None] * len(nulls)


def test_write_generation_metadata_records_the_gpu_and_the_split_file_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = types.ModuleType("torch")
    torch.__dict__["cuda"] = SimpleNamespace(get_device_name=lambda index: f"GPU {index}")
    monkeypatch.setitem(sys.modules, "torch", torch)
    split_file = tmp_path / "data" / "spider_data" / "dev.json"
    split_file.parent.mkdir(parents=True)
    split_file.write_text("[]")

    info = write(tmp_path)

    assert info["gpu"] == "GPU 0"
    assert info["split_file_sha256"] == hashlib.sha256(b"[]").hexdigest()


def test_git_is_none_when_git_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> str:
        raise FileNotFoundError("git")

    monkeypatch.setattr(meta.subprocess, "check_output", missing)

    assert meta.git("rev-parse", "HEAD") is None


def test_metadata_path_replaces_the_suffix(tmp_path: Path) -> None:
    assert meta.metadata_path(tmp_path / "a.jsonl") == tmp_path / "a.meta.json"
    assert meta.metadata_path(tmp_path / "a.report.json") == tmp_path / "a.report.meta.json"
