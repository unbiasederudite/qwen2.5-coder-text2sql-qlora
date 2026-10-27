"""Tests for the score command, text2sql.score."""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from conftest import QUESTIONS, fenced, make_test_split

nltk = pytest.importorskip("nltk")
if not nltk.download("punkt_tab", quiet=True):  # no-op when already downloaded
    pytest.skip("could not download the nltk tokenizer data", allow_module_level=True)

from text2sql.score import main
from text2sql.spider.predictions import Prediction


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs every test in its own directory, where the default `scores/` directory is created."""
    monkeypatch.chdir(tmp_path)


METADATA = {
    "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "model_revision": "abc123",
    "run": "baseline",
    "split": "dev",
    "generation": {"do_sample": False, "max_new_tokens": 256, "batch_size": 16},
}


def read_scores(tmp_path: Path, name: str = "run") -> tuple[dict[str, Any], dict[str, Any]]:
    """Reads the scores and the metadata of a report written to the default directory."""
    scores = json.loads((tmp_path / "scores" / f"{name}.report.json").read_text())
    metadata = json.loads((tmp_path / "scores" / f"{name}.report.meta.json").read_text())
    return scores, metadata


def predictions_file(
    tmp_path: Path, predictions: list[Prediction], metadata: dict[str, object] | None
) -> Path:
    """Writes `run.jsonl` and, if given, `run.meta.json` with the hash of the predictions."""
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    if metadata is not None:
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        (tmp_path / "run.meta.json").write_text(
            json.dumps({"predictions_sha256": sha256, **metadata})
        )
    return path


def run_main(
    path: Path, scoring_spider_dir: Path, suite_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    argv = [
        "score",
        str(path),
        "--spider-dir",
        str(scoring_spider_dir),
        "--test-suite-dir",
        str(suite_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    scores = json.loads((Path("scores") / f"{path.stem}.report.json").read_text())
    metadata = json.loads((Path("scores") / f"{path.stem}.report.meta.json").read_text())
    return {**scores, **metadata}


def test_main_prints_the_report_and_writes_it_as_json(
    scoring_spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    argv = [
        "score",
        str(path),
        "--spider-dir",
        str(scoring_spider_dir),
        "--test-suite-dir",
        str(suite_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    assert "TS" in capsys.readouterr().out
    scores, metadata = read_scores(tmp_path)
    assert scores["ts"]["all"] == 2 / 6
    assert metadata["split"] == "dev"


def test_main_asks_to_download_a_missing_test_suite(
    scoring_spider_dir: Path,
    predictions: list[Prediction],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    argv = [
        "score",
        str(path),
        "--spider-dir",
        str(scoring_spider_dir),
        "--test-suite-dir",
        str(tmp_path / "none"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        main()

    assert "download_data.sh test-suite" in capsys.readouterr().err


def test_main_scores_the_test_split_without_a_test_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spider = make_test_split(tmp_path / "spider", ["SELECT count(*) FROM singer"])
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps({"id": 0, "db_id": "concert", "predicted": "SELECT count(*) FROM singer"}) + "\n"
    )
    argv = [
        "score",
        str(path),
        "--split",
        "test",
        "--spider-dir",
        str(spider),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()  # the test split needs no test suite

    assert (
        "scores in scores/run.report.json, metadata in scores/run.report.meta.json"
        in capsys.readouterr().out
    )
    scores, metadata = read_scores(tmp_path)
    assert scores["ts"] is None
    assert metadata["split"] == "test"


def test_main_passes_keep_distinct_on_and_notes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spider = make_test_split(tmp_path / "spider", ["SELECT count(*) FROM singer"])
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps({"id": 0, "db_id": "concert", "predicted": "SELECT count(*) FROM singer"}) + "\n"
    )
    argv = ["score", str(path), "--split", "test", "--spider-dir", str(spider), "--keep-distinct"]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    assert "DISTINCT kept" in capsys.readouterr().out
    assert read_scores(tmp_path)[1]["keep_distinct"] is True


def test_main_raw_scores_the_replies_as_written(
    scoring_spider_dir: Path,
    suite_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": i, "db_id": "concert", "predicted": fenced(gold)}) + "\n"
            for i, (gold, _, _) in enumerate(QUESTIONS)
        )
    )
    argv = [
        "score",
        str(path),
        "--spider-dir",
        str(scoring_spider_dir),
        "--test-suite-dir",
        str(suite_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()
    assert "scored as written" not in capsys.readouterr().out
    first, first_metadata = read_scores(tmp_path)
    assert first["ex"]["all"] == 1.0
    assert first_metadata["predictions_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(sys, "argv", [*argv, "--raw"])
    main()
    assert "scored as written" in capsys.readouterr().out
    scores, metadata = read_scores(tmp_path)
    assert metadata["extraction"] is False
    assert scores["ex"]["all"] == 0.0


def test_main_reads_the_metadata_next_to_the_predictions(
    tmp_path: Path,
    scoring_spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = predictions_file(tmp_path, predictions, METADATA)

    report = run_main(path, scoring_spider_dir, suite_dir, monkeypatch)

    assert (report["model"], report["run"], report["model_revision"]) == (
        METADATA["model"],
        "baseline",
        "abc123",
    )


def test_main_scores_without_metadata_and_leaves_the_fields_empty(
    tmp_path: Path,
    scoring_spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = predictions_file(tmp_path, predictions, None)

    report = run_main(path, scoring_spider_dir, suite_dir, monkeypatch)

    assert report["model"] is None


def test_main_stops_when_the_metadata_belongs_to_another_predictions_file(
    tmp_path: Path,
    scoring_spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = predictions_file(tmp_path, predictions, METADATA)
    meta = json.loads((tmp_path / "run.meta.json").read_text())
    (tmp_path / "run.meta.json").write_text(json.dumps({**meta, "predictions_sha256": "0" * 64}))

    with pytest.raises(SystemExit):
        run_main(path, scoring_spider_dir, suite_dir, monkeypatch)

    assert "describes another predictions file" in capsys.readouterr().err
    assert not (tmp_path / "scores").exists()


def test_main_accepts_metadata_without_a_hash(
    tmp_path: Path,
    scoring_spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = predictions_file(tmp_path, predictions, METADATA)
    (tmp_path / "run.meta.json").write_text(json.dumps(METADATA))  # no predictions_sha256

    assert run_main(path, scoring_spider_dir, suite_dir, monkeypatch)["run"] == "baseline"


def test_main_warns_when_the_metadata_is_for_another_split(
    tmp_path: Path,
    scoring_spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = predictions_file(tmp_path, predictions, {**METADATA, "split": "test"})

    run_main(path, scoring_spider_dir, suite_dir, monkeypatch)

    assert "metadata is for the test split, scoring dev" in capsys.readouterr().err


def test_main_writes_the_metadata_next_to_the_scores_and_keeps_them_apart(
    tmp_path: Path,
    scoring_spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = predictions_file(tmp_path, predictions, None)
    run_main(path, scoring_spider_dir, suite_dir, monkeypatch)

    scores, metadata = read_scores(tmp_path)

    assert set(scores) == {"count", "ex", "ts", "em", "cm"}
    assert not set(scores) & set(metadata)
    assert metadata["predictions_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_main_reads_spider_from_the_default_folder(
    scoring_spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "spider_data").symlink_to(
        scoring_spider_dir
    )  # where the download script puts it
    path = predictions_file(tmp_path, predictions, None)
    monkeypatch.setattr(sys, "argv", ["score", str(path), "--test-suite-dir", str(suite_dir)])

    main()

    assert read_scores(tmp_path)[0]["count"]["all"] == len(predictions)
