"""Tests for text2sql.spider.database."""

from pathlib import Path

import pytest
from conftest import CONCERT_DDL, MUSEUM_DDL

from text2sql.spider.database import execution_error, load_schema_ddl


def test_load_schema_ddl_returns_statements_in_creation_order(spider_dir: Path) -> None:
    # The concert database uses AUTOINCREMENT, so SQLite also holds an internal sqlite_sequence table
    assert load_schema_ddl(spider_dir, "concert", "train_spider") == [f"{s};" for s in CONCERT_DDL]


def test_load_schema_ddl_test_reads_test_database(spider_dir: Path) -> None:
    assert load_schema_ddl(spider_dir, "museum", "test") == [f"{s};" for s in MUSEUM_DDL]


def test_load_schema_ddl_missing_database_raises(spider_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_schema_ddl(
            spider_dir, "museum", "train_spider"
        )  # test-only database, not in database/


def test_execution_error_is_none_for_valid_sql(spider_dir: Path) -> None:
    assert (
        execution_error(spider_dir, "concert", "SELECT count(*) FROM singer", "train_spider")
        is None
    )


def test_execution_error_returns_sqlite_message(spider_dir: Path) -> None:
    assert (
        execution_error(spider_dir, "concert", "SELECT name FROM band", "train_spider")
        == "no such table: band"
    )


def test_execution_error_cannot_modify_database(spider_dir: Path) -> None:
    error = execution_error(spider_dir, "concert", "DROP TABLE singer", "train_spider")

    assert error is not None and "readonly" in error
    assert len(load_schema_ddl(spider_dir, "concert", "train_spider")) == len(CONCERT_DDL)
