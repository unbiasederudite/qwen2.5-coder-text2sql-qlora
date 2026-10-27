"""Tests for text2sql.meta.runs."""

import hashlib
import json
import sqlite3
import types
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from text2sql.meta import provenance, runs
from text2sql.training import checkpoints

MODEL_CONFIG = {
    "tokenizer": {"padding_side": "left"},
    "generation": {"max_new_tokens": 256, "do_sample": False},
}
COUNTS = {"questions": 5, "generated": 5, "already_done": 0}
START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
CONFIG = {
    "data": {"splits": ["train_spider", "train_others"], "clean": True, "max_length": 2048},
    "lora": {"r": 16},
    "training": {"num_train_epochs": 2, "learning_rate": 2e-4},
}


def fake_model(quantization: object = None) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(name_or_path="some/model", quantization_config=quantization)
    )


def write(
    tmp_path: Path,
    model: object | None = None,
    finished_at: datetime = START + timedelta(seconds=90),
    adapter: str | None = None,
) -> dict[str, Any]:
    predictions = tmp_path / "predictions" / "m_run_dev.jsonl"
    predictions.parent.mkdir()
    predictions.write_text('{"id": 0}\n')
    model = model or fake_model()
    path = runs.write_generation_metadata(
        predictions,
        model,
        MODEL_CONFIG,
        "run",
        "dev",
        16,
        {"questions": 5, "generated": 3, "already_done": 2},
        START,
        finished_at,
        adapter,
    )
    assert path == predictions.with_suffix(".meta.json")
    result: dict[str, Any] = json.loads(path.read_text())
    return result


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def test_write_generation_metadata_records_the_run(tmp_path: Path) -> None:
    info = write(tmp_path)

    assert info["model"] == "some/model"
    assert (info["run"], info["split"]) == ("run", "dev")
    assert (info["questions"], info["already_done"]) == (5, 2)
    assert info["generation"] == {**MODEL_CONFIG["generation"], "batch_size": 16}
    assert info["tokenizer"] == MODEL_CONFIG["tokenizer"]
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
    assert "split_file_sha256" in capsys.readouterr().err


def test_write_generation_metadata_records_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance, "_git", lambda *args: "https://github.com/owner/name.git")

    assert write(tmp_path)["repo"] == "owner/name"


def test_write_generation_metadata_serializes_the_quantization_config(tmp_path: Path) -> None:
    class Config:
        def to_dict(self) -> dict[str, object]:
            return {"load_in_4bit": True}

    model = fake_model(Config())

    assert write(tmp_path, model)["quantization"] == {"load_in_4bit": True}


def test_write_generation_metadata_has_no_adapter_fields_without_an_adapter(tmp_path: Path) -> None:
    info = write(tmp_path)

    assert not set(provenance.ADAPTER_KEYS) & set(info)


def test_write_generation_metadata_hashes_the_weights_of_an_adapter_directory(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / checkpoints.ADAPTER_FILE).write_bytes(b"weights")

    info = write(tmp_path, adapter=str(adapter))

    assert info["adapter"] == str(adapter)
    assert info["adapter_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert "adapter_revision" not in info


def test_write_generation_metadata_records_the_commit_of_an_adapter_repository(
    tmp_path: Path, stub_module: Callable[..., types.ModuleType]
) -> None:
    stub_module(
        "huggingface_hub",
        try_to_load_from_cache=lambda repo, filename: f"/c/snapshots/def456/{filename}",
    )

    info = write(tmp_path, adapter="user/adapter")

    assert info["adapter"] == "user/adapter"
    assert info["adapter_revision"] == "def456"
    assert "adapter_sha256" not in info


def predictions_file(tmp_path: Path, info: dict[str, object] | None) -> Path:
    """Writes `run.jsonl` and, if given, `run.meta.json` with the hash of the predictions."""
    path = tmp_path / "run.jsonl"
    path.write_text('{"id": 0}\n')
    if info is not None:
        (tmp_path / "run.meta.json").write_text(
            json.dumps({"predictions_sha256": provenance.sha256(path), **info})
        )
    return path


def test_load_generation_metadata_reads_the_file_next_to_the_predictions(tmp_path: Path) -> None:
    assert runs.load_generation_metadata(predictions_file(tmp_path, None), "dev") is None

    info = runs.load_generation_metadata(predictions_file(tmp_path, {"run": "r"}), "dev")

    assert info is not None and info["run"] == "r"


def test_load_generation_metadata_stops_when_it_describes_another_predictions_file(
    tmp_path: Path,
) -> None:
    path = predictions_file(tmp_path, {})
    (tmp_path / "run.meta.json").write_text(json.dumps({"predictions_sha256": "0" * 64}))

    with pytest.raises(ValueError, match="describes another predictions file"):
        runs.load_generation_metadata(path, "dev")


def test_load_generation_metadata_accepts_metadata_without_a_hash(tmp_path: Path) -> None:
    path = predictions_file(tmp_path, {})
    (tmp_path / "run.meta.json").write_text(json.dumps({"run": "r"}))

    assert runs.load_generation_metadata(path, "dev") == {"run": "r"}


def test_load_generation_metadata_warns_about_another_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = predictions_file(tmp_path, {"split": "test"})

    runs.load_generation_metadata(path, "dev")

    assert "metadata is for the test split, scoring dev" in capsys.readouterr().err


def test_write_scoring_metadata_records_the_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = {
        ("remote", "get-url", "origin"): "git@github.com:owner/name.git",
        ("rev-parse", "HEAD"): "abc123",
    }
    monkeypatch.setattr(provenance, "_git", lambda *args: answers.get(args))
    given = {
        "model": "m",
        "model_revision": "r1",
        "run": "baseline",
        "generation": {"batch_size": 16},
    }
    predictions = predictions_file(tmp_path, None)

    path = runs.write_scoring_metadata(
        tmp_path / "x.report.json", predictions, "dev", True, False, given
    )

    assert path == tmp_path / "x.report.meta.json"
    assert json.loads(path.read_text()) == {
        "split": "dev",
        "predictions_sha256": provenance.sha256(predictions),
        "model": "m",
        "model_revision": "r1",
        "run": "baseline",  # the generation settings stay in the metadata of the predictions
        "keep_distinct": True,
        "extraction": False,
        "sqlite_version": sqlite3.sqlite_version,
        "repo": "owner/name",
        "commit": "abc123",
    }


def test_write_scoring_metadata_carries_the_adapter_over(tmp_path: Path) -> None:
    given = {"model": "m", "run": "finetuned", "adapter": "a/b", "adapter_revision": "r2"}
    predictions = predictions_file(tmp_path, None)

    path = runs.write_scoring_metadata(
        tmp_path / "x.report.json", predictions, "dev", True, False, given
    )

    info = json.loads(path.read_text())
    assert (info["adapter"], info["adapter_revision"]) == ("a/b", "r2")
    assert "adapter_sha256" not in info  # only what the predictions metadata has


def test_write_scoring_metadata_without_generation_metadata_or_git_has_nulls(
    tmp_path: Path,
) -> None:
    predictions = predictions_file(tmp_path, None)

    path = runs.write_scoring_metadata(
        tmp_path / "x.report.json", predictions, "dev", False, True, None
    )

    info = json.loads(path.read_text())  # tmp_path is no repository
    nulls = ("model", "model_revision", "run", "repo", "commit")
    assert [info[name] for name in nulls] == [None] * len(nulls)


def test_write_generation_metadata_records_the_gpu_and_the_split_file_hash(
    tmp_path: Path, stub_module: Callable[..., types.ModuleType]
) -> None:
    stub_module("torch", cuda=SimpleNamespace(get_device_name=lambda index: f"GPU {index}"))
    split_file = tmp_path / "data" / "spider_data" / "dev.json"
    split_file.parent.mkdir(parents=True)
    split_file.write_text("[]")

    info = write(tmp_path)

    assert info["gpu"] == "GPU 0"
    assert info["split_file_sha256"] == hashlib.sha256(b"[]").hexdigest()


def test_write_run_metadata_prints_the_fields_it_could_not_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs._write_run_metadata(tmp_path / "m.json", {"a": 1, "b": None, "c": None})

    assert "warning: fields that could not be read: b, c" in capsys.readouterr().err
    assert json.loads((tmp_path / "m.json").read_text()) == {"a": 1, "b": None, "c": None}


def write_training(
    tmp_path: Path, model: object | None = None, resumed: bool = False
) -> dict[str, Any]:
    path = runs.write_training_metadata(
        tmp_path,
        model or fake_model(),
        config=CONFIG,
        counts={"samples": 10, "dropped_too_long": 1},
        resumed=resumed,
        started_at=START,
        finished_at=START + timedelta(seconds=90),
    )
    assert path == tmp_path / "training.meta.json"
    result: dict[str, Any] = json.loads(path.read_text())
    return result


def test_write_training_metadata_records_the_run(tmp_path: Path) -> None:
    info = write_training(tmp_path)

    assert info["model"] == "some/model"
    assert info["data"] == CONFIG["data"]
    assert info["lora"] == CONFIG["lora"]
    assert info["training"] == CONFIG["training"]
    assert list(info["split_files_sha256"]) == CONFIG["data"]["splits"]
    assert (info["samples"], info["dropped_too_long"]) == (10, 1)
    assert info["duration_seconds"] == 90
    assert set(info["packages"]) == {"torch", "transformers", "bitsandbytes", "peft", "trl"}


@pytest.mark.parametrize("resumed", [False, True])
def test_write_training_metadata_records_whether_the_run_was_resumed(
    tmp_path: Path, resumed: bool
) -> None:
    assert write_training(tmp_path, resumed=resumed)["resumed"] is resumed


def test_write_training_metadata_serializes_the_quantization_config(tmp_path: Path) -> None:
    class Config:
        def to_dict(self) -> dict[str, object]:
            return {"load_in_4bit": True}

    assert write_training(tmp_path, fake_model(Config()))["quantization"] == {"load_in_4bit": True}


def test_warn_prints_to_stderr_with_a_prefix(capsys: pytest.CaptureFixture[str]) -> None:
    runs._warn("something is off")

    captured = capsys.readouterr()
    assert captured.err == "warning: something is off\n"
    assert captured.out == ""


def test_write_scoring_metadata_warns_about_the_fields_it_could_not_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions = predictions_file(tmp_path, None)

    runs.write_scoring_metadata(tmp_path / "x.report.json", predictions, "dev", False, True, None)

    err = capsys.readouterr().err
    assert "warning: fields that could not be read: run, model, model_revision" in err


def put_split_files(tmp_path: Path, **contents: str) -> None:
    """Writes `data/spider_data/<split>.json` files, which the metadata hashes."""
    directory = tmp_path / "data" / "spider_data"
    directory.mkdir(parents=True)
    for split, text in contents.items():
        (directory / f"{split}.json").write_text(text)


def test_write_generation_metadata_hashes_the_file_of_the_split_answered(tmp_path: Path) -> None:
    put_split_files(tmp_path, dev="[1]", test="[2]")
    predictions = tmp_path / "p.jsonl"
    predictions.write_text("")

    runs.write_generation_metadata(
        predictions, fake_model(), MODEL_CONFIG, "run", "test", 16, COUNTS, START, START
    )

    info = json.loads(runs.metadata_path(predictions).read_text())
    assert info["split_file_sha256"] == hashlib.sha256(b"[2]").hexdigest()


def test_write_training_metadata_hashes_the_file_of_each_training_split(tmp_path: Path) -> None:
    put_split_files(tmp_path, train_spider="[1]", train_others="[2]")

    info = write_training(tmp_path)

    assert info["split_files_sha256"] == {
        "train_spider": hashlib.sha256(b"[1]").hexdigest(),
        "train_others": hashlib.sha256(b"[2]").hexdigest(),
    }


def test_metadata_path_of_a_directory_with_dots_is_inside_it(tmp_path: Path) -> None:
    adapter = tmp_path / "qwen2.5-coder-1.5b-instruct"  # the dots must not be taken for a suffix
    adapter.mkdir()

    assert runs.metadata_path(adapter) == adapter / "training.meta.json"


def test_metadata_path_replaces_the_suffix(tmp_path: Path) -> None:
    assert runs.metadata_path(tmp_path / "a.jsonl") == tmp_path / "a.meta.json"
    assert runs.metadata_path(tmp_path / "a.report.json") == tmp_path / "a.report.meta.json"
