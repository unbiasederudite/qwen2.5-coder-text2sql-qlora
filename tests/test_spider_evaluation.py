"""Smoke tests for the vendored Spider evaluation code (see src/text2sql/spider_evaluation/README.md)."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

nltk = pytest.importorskip("nltk")
if not nltk.download("punkt_tab", quiet=True):  # no-op when already downloaded
    pytest.skip("could not download the nltk tokenizer data", allow_module_level=True)

from text2sql.spider_evaluation.evaluation import (
    Evaluator,
    build_valid_col_units,
    rebuild_sql_col,
    rebuild_sql_val,
)
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


def exact_match(db: Path, predicted: str, gold: str) -> tuple[bool, dict[str, dict[str, float]]]:
    """Exact set match and the component scores, prepared as `evaluate` does."""
    schema = Schema(get_schema(str(db)))
    evaluator = Evaluator()
    sqls = []
    for query in (predicted, gold):
        sql = rebuild_sql_val(get_sql(schema, query))
        sqls.append(
            rebuild_sql_col(build_valid_col_units(sql["from"]["table_units"], schema), sql, {})
        )
    return bool(evaluator.eval_exact_match(*sqls)), evaluator.partial_scores


GOLD_WHERE = "SELECT name FROM singer WHERE age > 35 AND name = 'Tim'"


def test_exact_match_ignores_the_order_of_conditions_and_of_columns(db: Path) -> None:
    swapped = "SELECT name FROM singer WHERE name = 'Tim' AND age > 35"

    assert exact_match(db, swapped, GOLD_WHERE)[0]
    assert exact_match(db, "SELECT age, name FROM singer", "SELECT name, age FROM singer")[0]


def test_exact_match_ignores_values_and_case(db: Path) -> None:
    other_value = "SELECT name FROM singer WHERE age > 99 AND name = 'Tim'"

    assert exact_match(db, other_value, GOLD_WHERE)[0]
    assert exact_match(db, GOLD_WHERE.lower(), GOLD_WHERE)[0]


def test_exact_match_rejects_a_different_condition(db: Path) -> None:
    assert not exact_match(
        db, "SELECT name FROM singer WHERE age < 35 AND name = 'Tim'", GOLD_WHERE
    )[0]


def test_component_matching_scores_each_clause_separately(db: Path) -> None:
    other_column = "SELECT age FROM singer WHERE age > 35 AND name = 'Tim'"
    other_operator = "SELECT name FROM singer WHERE age < 35 AND name = 'Tim'"

    _, wrong_select = exact_match(db, other_column, GOLD_WHERE)
    _, wrong_operator = exact_match(db, other_operator, GOLD_WHERE)

    assert (wrong_select["select"]["f1"], wrong_select["where"]["f1"]) == (0, 1)
    assert (wrong_operator["where"]["f1"], wrong_operator["where(no OP)"]["f1"]) == (0, 1)


@pytest.mark.parametrize(
    ("query", "level"),
    [
        ("SELECT count(*) FROM singer", "easy"),
        ("SELECT name FROM singer WHERE age > 35", "easy"),
        ("SELECT name FROM singer WHERE age > 35 ORDER BY age", "medium"),
        ("SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)", "hard"),
        (
            "SELECT name FROM singer WHERE age > 40 UNION SELECT name FROM singer WHERE age < 30",
            "hard",
        ),
        ("SELECT name FROM singer WHERE age > 35 AND name LIKE 'T%' ORDER BY age LIMIT 1", "extra"),
    ],
)
def test_hardness_levels(db: Path, query: str, level: str) -> None:
    sql = get_sql(Schema(get_schema(str(db))), query)

    assert Evaluator().eval_hardness(sql) == level
