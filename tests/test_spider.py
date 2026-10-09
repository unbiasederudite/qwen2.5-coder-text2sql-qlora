from pathlib import Path

import pytest
from conftest import CONCERT_DDL, MUSEUM_DDL, TEST, TRAIN

from text2sql.spider import (
    SPIDER_DIR,
    load_examples,
    load_schema_ddl,
    load_schemas,
    query_error,
    split_file,
)


def test_split_file_names_the_questions_file_of_a_split() -> None:
    assert split_file(Path("spider"), "dev") == Path("spider/dev.json")
    assert split_file(SPIDER_DIR, "train_others") == Path("data/spider_data/train_others.json")


def test_load_examples_keeps_three_fields_in_file_order(spider_dir: Path) -> None:
    assert load_examples(spider_dir, "train_spider") == TRAIN


def test_load_examples_reads_the_requested_split(spider_dir: Path) -> None:
    assert load_examples(spider_dir, "test") == TEST


def test_load_schemas_keys_by_db_id(spider_dir: Path) -> None:
    schemas = load_schemas(spider_dir)

    assert set(schemas) == {"concert", "shop"}
    assert schemas["concert"]["table_names_original"] == ["singer", "concert"]
    assert "table_names" not in schemas["concert"]


def test_load_schemas_test_reads_test_tables(spider_dir: Path) -> None:
    assert set(load_schemas(spider_dir, test=True)) == {"museum"}


def test_load_schema_ddl_returns_statements_in_creation_order(spider_dir: Path) -> None:
    # The concert database uses AUTOINCREMENT, so SQLite also holds an internal sqlite_sequence table
    assert load_schema_ddl(spider_dir, "concert") == [f"{s};" for s in CONCERT_DDL]


def test_load_schema_ddl_test_reads_test_database(spider_dir: Path) -> None:
    assert load_schema_ddl(spider_dir, "museum", test=True) == [f"{s};" for s in MUSEUM_DDL]


def test_load_schema_ddl_missing_database_raises(spider_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_schema_ddl(spider_dir, "museum")  # test-only database, not in database/


def test_query_error_is_none_for_valid_sql(spider_dir: Path) -> None:
    assert query_error(spider_dir, "concert", "SELECT count(*) FROM singer") is None


def test_query_error_returns_sqlite_message(spider_dir: Path) -> None:
    assert query_error(spider_dir, "concert", "SELECT name FROM band") == "no such table: band"


def test_query_error_cannot_modify_database(spider_dir: Path) -> None:
    error = query_error(spider_dir, "concert", "DROP TABLE singer")

    assert error is not None and "readonly" in error
    assert len(load_schema_ddl(spider_dir, "concert")) == len(CONCERT_DDL)
