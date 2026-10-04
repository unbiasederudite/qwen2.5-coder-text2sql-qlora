"""Smoke tests for the vendored Spider evaluation code (see src/text2sql/spider_evaluation/README.md)."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

nltk = pytest.importorskip("nltk")
if not nltk.download("punkt_tab", quiet=True):  # no-op when already downloaded
    pytest.skip("could not download the nltk tokenizer data", allow_module_level=True)

from text2sql.spider_evaluation.evaluation import Evaluator
from text2sql.spider_evaluation.exec_eval import eval_exec_match, exec_on_db
from text2sql.spider_evaluation.process_sql import Schema, get_schema, get_sql


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """A tiny database in the layout the official code expects: <db_id>/<db_id>.sqlite."""
    path = tmp_path / "concert" / "concert.sqlite"
    path.parent.mkdir()
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    con.executemany(
        "INSERT INTO singer VALUES (?, ?, ?)",
        [(1, "Joe", 52), (2, "Tim", 32), (3, "Tim", 32)],
    )
    con.commit()
    con.close()
    return path


def matches(db: Path, predicted: str, gold: str) -> bool:
    return bool(eval_exec_match(str(db), predicted, gold, False, False, False))


def test_parser_reads_a_query_and_the_hardness_is_easy(db: Path) -> None:
    sql = get_sql(Schema(get_schema(str(db))), "SELECT count(*) FROM singer")

    assert {"select", "from", "where", "orderBy"} <= set(sql)
    assert Evaluator().eval_hardness(sql) == "easy"


def test_a_differently_written_query_with_the_same_result_matches(db: Path) -> None:
    assert matches(db, "SELECT COUNT(singer_id) FROM singer", "SELECT count(*) FROM singer")


def test_a_query_with_a_different_result_does_not_match(db: Path) -> None:
    assert not matches(
        db, "SELECT count(*) FROM singer WHERE age > 40", "SELECT count(*) FROM singer"
    )


def test_a_query_that_fails_does_not_match(db: Path) -> None:
    assert not matches(db, "SELECT count(*) FROM singers", "SELECT count(*) FROM singer")


def test_row_order_is_ignored_unless_the_gold_query_orders(db: Path) -> None:
    assert matches(db, "SELECT name FROM singer ORDER BY age", "SELECT name FROM singer")
    assert not matches(
        db, "SELECT name FROM singer ORDER BY age", "SELECT name FROM singer ORDER BY age DESC"
    )


def test_column_order_is_ignored(db: Path) -> None:
    assert matches(db, "SELECT age, name FROM singer", "SELECT name, age FROM singer")


def test_distinct_is_ignored_by_default(db: Path) -> None:
    assert matches(db, "SELECT DISTINCT name, age FROM singer", "SELECT name, age FROM singer")


def test_duplicate_rows_count(db: Path) -> None:
    two_rows = "SELECT name FROM singer WHERE singer_id IN (2, 3)"  # Tim, Tim

    assert not matches(db, two_rows, "SELECT name FROM singer WHERE singer_id = 2")  # Tim


def test_a_prediction_cannot_modify_the_database(db: Path) -> None:
    assert not matches(db, "DELETE FROM singer", "SELECT count(*) FROM singer")

    count = sqlite3.connect(db).execute("SELECT count(*) FROM singer").fetchone()
    assert count == (3,)


def test_a_runaway_query_is_stopped(db: Path) -> None:
    loop = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r"

    flag, _ = asyncio.run(exec_on_db(str(db), loop, timeout=0.5))

    assert flag == "exception"


def test_a_missing_database_is_not_created(tmp_path: Path) -> None:
    flag, _ = asyncio.run(exec_on_db(str(tmp_path / "none.sqlite"), "SELECT 1"))

    assert flag == "exception"
    assert not (tmp_path / "none.sqlite").exists()
