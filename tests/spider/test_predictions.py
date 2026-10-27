"""Tests for text2sql.spider.predictions."""

from pathlib import Path

import pytest

from text2sql.spider.predictions import Prediction, append_prediction, load_predictions


def test_load_predictions_reads_json_lines_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text(
        '{"id": 0, "db_id": "a", "predicted": "SELECT 1"}\n\n{"id": 1, "db_id": "b", "predicted": ""}\n'
    )

    assert load_predictions(path) == [
        {"id": 0, "db_id": "a", "predicted": "SELECT 1"},
        {"id": 1, "db_id": "b", "predicted": ""},
    ]


def test_load_predictions_names_the_line_with_a_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text('{"id": 0, "db_id": "a", "predicted": "x"}\n{"id": 1, "db_id": "a"}\n')

    with pytest.raises(ValueError, match=r"p\.jsonl:2: missing field 'predicted'"):
        load_predictions(path)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "invalid JSON"),
        ("[1, 2]", "not a JSON object"),
        ('{"id": "0", "db_id": "a", "predicted": "x"}', "id must be an integer"),
        ('{"id": true, "db_id": "a", "predicted": "x"}', "id must be an integer"),
        ('{"id": 0, "db_id": "a", "predicted": null}', "predicted strings"),
    ],
)
def test_load_predictions_rejects_a_bad_line_and_names_it(
    tmp_path: Path, line: str, message: str
) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text('{"id": 0, "db_id": "a", "predicted": "x"}\n' + line + "\n")

    with pytest.raises(ValueError, match=rf"p\.jsonl:2: .*{message}"):
        load_predictions(path)


def test_append_prediction_writes_lines_that_load_predictions_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    first = Prediction(id=0, db_id="a", predicted="SELECT 1")
    second = Prediction(id=3, db_id="b", predicted="SELECT 2")

    with path.open("a") as file:
        append_prediction(file, first)
    with path.open("a") as file:
        append_prediction(file, second)

    assert load_predictions(path) == [first, second]
