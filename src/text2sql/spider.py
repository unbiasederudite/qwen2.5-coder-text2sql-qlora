"""Loaders for the Spider dataset."""

import json
import sqlite3
from pathlib import Path
from typing import Literal, TypedDict

Split = Literal["train_spider", "train_others", "dev", "test"]


class Example(TypedDict):
    """One Spider question with its gold SQL."""

    db_id: str  # database name
    question: str  # natural-language question
    query: str  # gold SQL


class Schema(TypedDict):
    """Structure of one Spider database, as stored in `tables.json`."""

    db_id: str  # database name
    table_names_original: list[str]  # table names
    column_names_original: list[list[int | str]]  # [table_index, column_name] pairs
    column_types: list[str]  # column types
    primary_keys: list[int | list[int]]  # primary key column indices
    foreign_keys: list[list[int]]  # [column_index, referenced_column_index] pairs


def load_examples(spider_dir: Path, split: Split) -> list[Example]:
    """Loads the questions and gold SQL of one split.

    Args:
        spider_dir (Path): Spider directory.
        split (Split): Split to load.

    Returns:
        list[Example]: Examples in file order.
    """
    rows = json.loads((spider_dir / f"{split}.json").read_text())
    return [Example(db_id=r["db_id"], question=r["question"], query=r["query"]) for r in rows]


def load_schemas(spider_dir: Path, test: bool = False) -> dict[str, Schema]:
    """Loads the database schemas from `tables.json`.

    Args:
        spider_dir (Path): Spider directory.
        test (bool): Read `test_tables.json` instead.

    Returns:
        dict[str, Schema]: Schemas by `db_id`.
    """
    name = "test_tables.json" if test else "tables.json"
    rows = json.loads((spider_dir / name).read_text())
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


def database_path(spider_dir: Path, db_id: str, test: bool = False) -> Path:
    """Returns the path of one database's SQLite file.

    Args:
        spider_dir (Path): Spider directory.
        db_id (str): Database name.
        test (bool): Look in `test_database/` instead.

    Returns:
        Path: Path of the `.sqlite` file.

    Raises:
        FileNotFoundError: If the database file is missing.
    """
    folder = "test_database" if test else "database"
    path = spider_dir / folder / db_id / f"{db_id}.sqlite"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_schema_ddl(spider_dir: Path, db_id: str, test: bool = False) -> list[str]:
    """Reads the `CREATE TABLE` statements of one database.

    Args:
        spider_dir (Path): Spider directory.
        db_id (str): Database name.
        test (bool): Read from `test_database/` instead.

    Returns:
        list[str]: One `CREATE TABLE` statement per table.

    Raises:
        FileNotFoundError: If the database file is missing.
    """
    path = database_path(spider_dir, db_id, test=test)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY rowid"
        ).fetchall()
    finally:
        con.close()
    return [f"{sql};" for (sql,) in rows]


def query_error(spider_dir: Path, db_id: str, query: str, test: bool = False) -> str | None:
    """Runs a query on one database.

    Args:
        spider_dir (Path): Spider directory.
        db_id (str): Database name.
        query (str): SQL to run.
        test (bool): Read from `test_database/` instead.

    Returns:
        str | None: SQLite error message, or None if the query runs.

    Raises:
        FileNotFoundError: If the database file is missing.
    """
    path = database_path(spider_dir, db_id, test=test)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        con.execute(query).fetchone()
    except sqlite3.Error as e:
        return str(e)
    finally:
        con.close()
    return None
