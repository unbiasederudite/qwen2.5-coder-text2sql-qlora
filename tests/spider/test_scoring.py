"""Tests for text2sql.spider.scoring."""

import json
import sqlite3
from pathlib import Path

import pytest
from conftest import QUESTIONS, fenced, make_test_split

nltk = pytest.importorskip("nltk")
if not nltk.download("punkt_tab", quiet=True):  # no-op when already downloaded
    pytest.skip("could not download the nltk tokenizer data", allow_module_level=True)

from text2sql.spider import scoring
from text2sql.spider.dataset import Example
from text2sql.spider.predictions import Prediction
from text2sql.spider.report import format_report
from text2sql.spider.scoring import score

LEVELS = ["easy", "medium", "hard", "extra", "all"]


EXAMPLES = [
    Example(db_id="a", question="?", query="SELECT 1"),
    Example(db_id="b", question="?", query="SELECT 2"),
]


def test_align_orders_the_predictions_by_id() -> None:
    shuffled = [
        Prediction(id=1, db_id="b", predicted="two"),
        Prediction(id=0, db_id="a", predicted="one"),
    ]

    assert scoring._align(shuffled, EXAMPLES) == ["one", "two"]


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        ([0, 0, 1], "duplicate id 0"),
        ([0], "missing \\[1\\]"),
        ([0, 1, 2], "unknown \\[2\\]"),
    ],
)
def test_align_rejects_ids_that_do_not_match_the_questions(ids: list[int], message: str) -> None:
    db_ids = {0: "a", 1: "b", 2: "c"}
    predictions = [Prediction(id=i, db_id=db_ids[i], predicted="x") for i in ids]

    with pytest.raises(ValueError, match=message):
        scoring._align(predictions, EXAMPLES)


def test_align_rejects_a_prediction_for_another_database() -> None:
    predictions = [
        Prediction(id=0, db_id="a", predicted="x"),
        Prediction(id=1, db_id="a", predicted="x"),
    ]

    with pytest.raises(ValueError, match="id 1: db_id a is not b"):
        scoring._align(predictions, EXAMPLES)


def test_score_counts_the_questions_by_difficulty(
    scoring_spider_dir: Path, suite_dir: Path, predictions: list[Prediction]
) -> None:
    report = score(predictions, scoring_spider_dir, "dev", suite_dir)

    assert report["count"] == {"easy": 2, "medium": 1, "hard": 1, "extra": 2, "all": 6}


def test_score_execution_accuracy_uses_the_original_database(
    scoring_spider_dir: Path, suite_dir: Path, predictions: list[Prediction]
) -> None:
    report = score(predictions, scoring_spider_dir, "dev", suite_dir)

    # `SELECT 3` and `age > 40` happen to be right on the original database
    assert report["ex"] == {"easy": 1.0, "medium": 1.0, "hard": 1.0, "extra": 0.0, "all": 4 / 6}


def test_score_test_suite_accuracy_catches_right_by_luck(
    scoring_spider_dir: Path, suite_dir: Path, predictions: list[Prediction]
) -> None:
    report = score(predictions, scoring_spider_dir, "dev", suite_dir)

    assert report["ts"] == {"easy": 0.5, "medium": 1.0, "hard": 0.0, "extra": 0.0, "all": 2 / 6}


def test_score_exact_match(
    scoring_spider_dir: Path, suite_dir: Path, predictions: list[Prediction]
) -> None:
    report = score(predictions, scoring_spider_dir, "dev", suite_dir)

    assert report["em"] == {"easy": 0.5, "medium": 1.0, "hard": 0.0, "extra": 0.0, "all": 2 / 6}


def test_score_component_matching(
    scoring_spider_dir: Path, suite_dir: Path, predictions: list[Prediction]
) -> None:
    cm = score(predictions, scoring_spider_dir, "dev", suite_dir)["cm"]

    assert cm["medium"]["where"] == {"acc": 1.0, "rec": 1.0, "f1": 1.0}
    assert cm["medium"]["order"] == {"acc": 1.0, "rec": 1.0, "f1": 1.0}
    assert cm["hard"]["select"]["acc"] == 1.0  # both select `name`
    assert cm["hard"]["where"]["acc"] == 0.0  # `age > 40` is not `age > (SELECT ...)`


def test_score_without_the_test_suite_has_no_test_suite_accuracy(
    scoring_spider_dir: Path, predictions: list[Prediction]
) -> None:
    report = score(predictions, scoring_spider_dir, "dev")

    assert report["ts"] is None
    assert report["ex"]["all"] == 4 / 6
    assert report["em"]["all"] == 2 / 6


def test_score_fails_when_the_tokenizer_data_cannot_be_downloaded(
    scoring_spider_dir: Path, predictions: list[Prediction], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scoring.nltk, "download", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="tokenizer data"):
        score(predictions, scoring_spider_dir, "dev")


def test_format_report_shows_a_dash_without_the_test_suite(
    scoring_spider_dir: Path, predictions: list[Prediction]
) -> None:
    lines = format_report(score(predictions, scoring_spider_dir, "dev")).splitlines()

    assert lines[0].split() == LEVELS
    assert [line.split()[0] for line in lines[1:]] == ["count", "EX", "TS", "EM"]
    assert lines[3].split() == ["TS", "-", "-", "-", "-", "-"]


def test_a_prediction_containing_value_is_not_rewritten(tmp_path: Path) -> None:
    spider = tmp_path / "spider"
    (spider / "database" / "shop").mkdir(parents=True)
    con = sqlite3.connect(spider / "database" / "shop" / "shop.sqlite")
    con.execute("CREATE TABLE item (item_id int PRIMARY KEY, value int)")
    con.execute("INSERT INTO item VALUES (1, 10), (2, 20)")
    con.commit()
    con.close()
    query = "SELECT max(value) FROM item"
    (spider / "dev.json").write_text(
        json.dumps([{"db_id": "shop", "question": "?", "query": query}])
    )
    (spider / "tables.json").write_text(
        json.dumps(
            [
                {
                    "db_id": "shop",
                    "table_names_original": ["item"],
                    "column_names_original": [[-1, "*"], [0, "item_id"], [0, "value"]],
                    "column_types": ["text", "number", "number"],
                    "primary_keys": [1],
                    "foreign_keys": [],
                }
            ]
        )
    )

    report = score([Prediction(id=0, db_id="shop", predicted=query)], spider, "dev")

    assert report["ex"]["all"] == 1.0
    assert report["em"]["all"] == 1.0


def test_score_on_the_test_split_reads_the_test_files_and_has_no_test_suite(tmp_path: Path) -> None:
    spider = make_test_split(tmp_path, ["SELECT count(*) FROM singer", "SELECT name FROM singer"])
    predictions = [
        Prediction(id=0, db_id="concert", predicted="SELECT count(*) FROM singer"),
        Prediction(id=1, db_id="concert", predicted="SELECT age FROM singer"),
    ]

    report = score(predictions, spider, "test")

    assert report["count"]["all"] == 2
    assert report["ts"] is None
    assert report["ex"]["all"] == 0.5
    assert report["em"]["all"] == 0.5


def test_score_collapses_whitespace_in_the_gold_and_the_prediction(tmp_path: Path) -> None:
    # Two Spider test gold queries contain a newline or a tab, which the official file format cannot hold
    spider = make_test_split(tmp_path, ["SELECT count(*)\n\tFROM singer"])
    predictions = [Prediction(id=0, db_id="concert", predicted="SELECT   count(*)\nFROM\tsinger\n")]

    report = score(predictions, spider, "test")

    assert report["ex"]["all"] == 1.0
    assert report["em"]["all"] == 1.0


def test_distinct_is_stripped_from_execution_unless_kept(tmp_path: Path) -> None:
    spider = make_test_split(tmp_path, ["SELECT count(DISTINCT name) FROM singer"])
    con = sqlite3.connect(spider / "test_database" / "concert" / "concert.sqlite")
    con.execute("INSERT INTO singer VALUES (4, 'Tim', 33)")  # two Tims: 3 distinct names, 4 names
    con.commit()
    con.close()
    predictions = [Prediction(id=0, db_id="concert", predicted="SELECT count(name) FROM singer")]

    stripped = score(predictions, spider, "test")
    kept = score(predictions, spider, "test", keep_distinct=True)

    assert (stripped["ex"]["all"], kept["ex"]["all"]) == (1.0, 0.0)
    assert stripped["em"]["all"] == kept["em"]["all"] == 1.0  # EM ignores DISTINCT either way


def test_a_value_with_repeated_spaces_is_matched_as_it_is(tmp_path: Path) -> None:
    gold = "SELECT name FROM singer WHERE name = 'Tim  Lee'"
    spider = make_test_split(tmp_path, [gold])
    con = sqlite3.connect(spider / "test_database" / "concert" / "concert.sqlite")
    con.execute("INSERT INTO singer VALUES (5, 'Tim  Lee', 30)")
    con.commit()
    con.close()
    right = [Prediction(id=0, db_id="concert", predicted=gold)]
    # Empty like the gold would be if its two spaces were collapsed
    wrong = [Prediction(id=0, db_id="concert", predicted="SELECT name FROM singer WHERE age = 99")]

    assert score(right, spider, "test")["ex"]["all"] == 1.0
    assert score(wrong, spider, "test")["ex"]["all"] == 0.0


def test_score_takes_the_sql_out_of_the_replies_by_default(
    scoring_spider_dir: Path, suite_dir: Path
) -> None:
    predictions = [
        Prediction(id=i, db_id="concert", predicted=fenced(gold))
        for i, (gold, _, _) in enumerate(QUESTIONS)
    ]

    report = score(predictions, scoring_spider_dir, "dev", suite_dir)

    assert report["ex"]["all"] == report["ts"]["all"] == report["em"]["all"] == 1.0  # type: ignore[index]


def test_score_can_score_the_replies_as_written(scoring_spider_dir: Path, suite_dir: Path) -> None:
    predictions = [
        Prediction(id=i, db_id="concert", predicted=fenced(gold))
        for i, (gold, _, _) in enumerate(QUESTIONS)
    ]

    report = score(predictions, scoring_spider_dir, "dev", suite_dir, extract=False)

    assert report["ex"]["all"] == report["em"]["all"] == 0.0  # the fence is not SQL
