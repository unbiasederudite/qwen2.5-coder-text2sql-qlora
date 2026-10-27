"""Spider files: splits, questions and schemas."""

import json
from pathlib import Path
from typing import Literal, TypedDict

Split = Literal["train_spider", "train_others", "dev", "test"]
TRAIN_SPLITS: tuple[Split, ...] = ("train_spider", "train_others")  # dev and test are held out
SPIDER_DIR = Path("data/spider_data")  # where `scripts/download_data.sh spider` puts Spider
TEST_SUITE_DIR = Path("data/test_suite_data")  # where `scripts/download_data.sh test-suite` puts it


class Example(TypedDict):
    """One Spider question with its gold SQL."""

    db_id: str  # database name
    question: str  # natural-language question
    query: str  # gold SQL


class Schema(TypedDict):
    """One Spider database schema, as stored in `tables.json`."""

    db_id: str  # database name
    table_names_original: list[str]  # table names
    column_names_original: list[list[int | str]]  # [table_index, column_name] pairs
    column_types: list[str]  # column types
    primary_keys: list[int | list[int]]  # primary key column indices
    foreign_keys: list[list[int]]  # [column_index, referenced_column_index] pairs


def split_file(spider_dir: Path, split: Split) -> Path:
    """Names the questions file of one split.

    Args:
        spider_dir (Path): Spider directory.
        split (Split): Split to name.

    Returns:
        Path: JSON file of the split.
    """
    return spider_dir / f"{split}.json"


def tables_file(spider_dir: Path, split: Split) -> Path:
    """Names the schemas file of the databases of one split.

    Args:
        spider_dir (Path): Spider directory.
        split (Split): Split whose databases are wanted. The test split has its own file.

    Returns:
        Path: JSON file of the schemas.
    """
    return spider_dir / ("test_tables.json" if split == "test" else "tables.json")


def database_dir(spider_dir: Path, split: Split) -> Path:
    """Names the directory of the databases of one split.

    Args:
        spider_dir (Path): Spider directory.
        split (Split): Split whose databases are wanted. The test split has its own directory.

    Returns:
        Path: Directory with one folder per database.
    """
    return spider_dir / ("test_database" if split == "test" else "database")


def load_examples(spider_dir: Path, split: Split) -> list[Example]:
    """Loads the questions and gold SQL of one split.

    Args:
        spider_dir (Path): Spider directory.
        split (Split): Split to load.

    Returns:
        list[Example]: Examples in file order.
    """
    rows = json.loads(split_file(spider_dir, split).read_text())
    return [Example(db_id=r["db_id"], question=r["question"], query=r["query"]) for r in rows]


def load_schemas(spider_dir: Path, split: Split) -> dict[str, Schema]:
    """Loads the schemas of the databases of one split.

    Args:
        spider_dir (Path): Spider directory.
        split (Split): Split whose databases are wanted.

    Returns:
        dict[str, Schema]: Schemas by `db_id`.
    """
    rows = json.loads(tables_file(spider_dir, split).read_text())
    return {
        r["db_id"]: Schema(
            db_id=r["db_id"],
            table_names_original=r["table_names_original"],
            column_names_original=r["column_names_original"],
            column_types=r["column_types"],
            primary_keys=r["primary_keys"],
            foreign_keys=r["foreign_keys"],
        )
        for r in rows
    }
