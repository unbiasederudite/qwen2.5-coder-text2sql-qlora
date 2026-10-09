import importlib
import json
import sqlite3
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import text2sql

CONCERT_DDL = [
    "CREATE TABLE singer (singer_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)",
    (
        "CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, singer_id INTEGER, "
        "FOREIGN KEY (singer_id) REFERENCES singer (singer_id))"
    ),
]
SHOP_DDL = ["CREATE TABLE shop (shop_id INTEGER PRIMARY KEY, name TEXT)"]
MUSEUM_DDL = ["CREATE TABLE museum (museum_id INTEGER PRIMARY KEY, name TEXT)"]

TRAIN = [
    {
        "db_id": "concert",
        "question": "How many singers are there?",
        "query": "SELECT count(*) FROM singer",
    },
    {
        "db_id": "concert",
        "question": "How many singers are there?",
        "query": "SELECT count(*) FROM singer",
    },
    {"db_id": "concert", "question": "List the bands.", "query": "SELECT name FROM band"},
    {"db_id": "shop", "question": "List the shop names.", "query": "SELECT name FROM shop"},
]
TEST = [{"db_id": "museum", "question": "List the museums.", "query": "SELECT name FROM museum"}]


def make_database(path: Path, ddl: list[str]) -> None:
    path.parent.mkdir(parents=True)
    con = sqlite3.connect(path)
    for statement in ddl:
        con.execute(statement)
    con.commit()
    con.close()


def schema_row(db_id: str, tables: list[str]) -> dict[str, object]:
    return {
        "db_id": db_id,
        "table_names_original": tables,
        "column_names_original": [[-1, "*"]] + [[i, "name"] for i in range(len(tables))],
        "column_types": ["text"] * (len(tables) + 1),
        "primary_keys": [],
        "foreign_keys": [],
        "table_names": tables,  # extra field the loader drops
    }


@pytest.fixture
def spider_dir(tmp_path: Path) -> Path:
    """A miniature Spider release: two train databases and one test-only database."""
    # Extra fields, like the real release, which the loader drops
    for name, rows in (("train_spider", TRAIN), ("test", TEST)):
        rows = [{**r, "query_toks": r["query"].split(), "sql": {}} for r in rows]
        (tmp_path / f"{name}.json").write_text(json.dumps(rows))

    (tmp_path / "tables.json").write_text(
        json.dumps([schema_row("concert", ["singer", "concert"]), schema_row("shop", ["shop"])])
    )
    (tmp_path / "test_tables.json").write_text(json.dumps([schema_row("museum", ["museum"])]))

    make_database(tmp_path / "database" / "concert" / "concert.sqlite", CONCERT_DDL)
    make_database(tmp_path / "database" / "shop" / "shop.sqlite", SHOP_DDL)
    make_database(tmp_path / "test_database" / "museum" / "museum.sqlite", MUSEUM_DDL)
    return tmp_path


@pytest.fixture
def stub_module(monkeypatch: pytest.MonkeyPatch) -> Callable[..., types.ModuleType]:
    """Installs a stand-in for a module that needs a GPU or an install, for one test."""

    def install(name: str, **attributes: object) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    return install


@pytest.fixture
def fresh_import(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], types.ModuleType]]:
    """Imports a `text2sql` module anew, on the stand-ins installed before, and drops it after."""
    imported: list[str] = []

    def load(name: str) -> types.ModuleType:
        monkeypatch.delitem(sys.modules, f"text2sql.{name}", raising=False)
        monkeypatch.delattr(text2sql, name, raising=False)
        imported.append(name)
        return importlib.import_module(f"text2sql.{name}")

    yield load
    for name in imported:  # the module was built on the stand-ins, so do not leak it
        sys.modules.pop(f"text2sql.{name}", None)
        text2sql.__dict__.pop(name, None)
