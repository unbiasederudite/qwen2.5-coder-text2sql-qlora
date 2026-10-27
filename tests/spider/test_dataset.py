"""Tests for text2sql.spider.dataset."""

from pathlib import Path

from conftest import TEST, TRAIN

from text2sql.spider.dataset import (
    SPIDER_DIR,
    database_dir,
    load_examples,
    load_schemas,
    split_file,
    tables_file,
)


def test_split_file_names_the_questions_file_of_a_split() -> None:
    assert split_file(Path("spider"), "dev") == Path("spider/dev.json")
    assert split_file(SPIDER_DIR, "train_others") == Path("data/spider_data/train_others.json")


def test_load_examples_keeps_three_fields_in_file_order(spider_dir: Path) -> None:
    assert load_examples(spider_dir, "train_spider") == TRAIN


def test_load_examples_reads_the_requested_split(spider_dir: Path) -> None:
    assert load_examples(spider_dir, "test") == TEST


def test_load_schemas_keys_by_db_id(spider_dir: Path) -> None:
    schemas = load_schemas(spider_dir, "train_spider")

    assert set(schemas) == {"concert", "shop"}
    assert schemas["concert"]["table_names_original"] == ["singer", "concert"]
    assert "table_names" not in schemas["concert"]


def test_load_schemas_test_reads_test_tables(spider_dir: Path) -> None:
    assert set(load_schemas(spider_dir, "test")) == {"museum"}


def test_tables_file_and_database_dir_depend_on_the_split_group() -> None:
    root = Path("spider")

    assert tables_file(root, "dev") == root / "tables.json"
    assert tables_file(root, "test") == root / "test_tables.json"
    assert database_dir(root, "dev") == root / "database"
    assert database_dir(root, "test") == root / "test_database"
