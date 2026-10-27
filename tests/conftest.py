"""Test data, fixtures and stand-ins shared by the tests."""

import importlib
import json
import sqlite3
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from text2sql.spider.predictions import Prediction
from text2sql.spider.samples import Sample

PROJECT_ROOT = Path(__file__).parent.parent

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
    """Imports a `text2sql` module anew, on the stand-ins installed before, and drops it after.

    The name is relative to `text2sql` and may be dotted, as in `training.train`.
    """
    imported: list[str] = []

    def load(name: str) -> types.ModuleType:
        full = f"text2sql.{name}"
        parent_name, _, leaf = full.rpartition(".")
        monkeypatch.delitem(sys.modules, full, raising=False)
        monkeypatch.delattr(importlib.import_module(parent_name), leaf, raising=False)
        imported.append(full)
        return importlib.import_module(full)

    yield load
    for full in imported:  # the module was built on the stand-ins, so do not leak it
        parent_name, _, leaf = full.rpartition(".")
        sys.modules.pop(full, None)
        sys.modules[parent_name].__dict__.pop(leaf, None)


TRAIN_CONFIG = {
    "data": {
        "splits": ["train_spider"],
        "clean": True,
        "max_length": 50,  # the stand-in tokenizer counts a word as a token
    },
    "lora": {"r": 16, "lora_alpha": 32},
    "training": {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 4,
        "learning_rate": 2e-4,
        "seed": 7,
    },
}


class FakeTokenizer:
    """Counts a word as a token."""

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool) -> str:
        assert tokenize is False
        return " ".join(message["content"] for message in messages)

    def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[str]]:
        assert add_special_tokens is False
        return {"input_ids": text.split()}


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    @classmethod
    def from_list(cls, rows: list[dict[str, Any]]) -> "FakeDataset":
        return cls(rows)

    def __len__(self) -> int:
        return len(self.rows)


def make_sample(schema_words: int = 1, query: str = "SELECT 1") -> Sample:
    return Sample(
        db_id="db",
        question="how many?",
        query=query,
        schema=[" ".join(["CREATE"] * schema_words)],
    )


# The miniature release and predictions that the scoring tests and the score command tests share
TABLES = [
    {
        "db_id": "concert",
        "table_names_original": ["singer"],
        "column_names_original": [[-1, "*"], [0, "singer_id"], [0, "name"], [0, "age"]],
        "column_types": ["text", "number", "text", "number"],
        "primary_keys": [1],
        "foreign_keys": [],
    }
]


ALL_AGE = "SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)"


QUESTIONS = [
    # Same query: right everywhere
    ("SELECT count(*) FROM singer", "SELECT count(*) FROM singer", "easy"),
    # Right only by luck: 3 singers in the original database but 4 in the other version
    ("SELECT count(*) FROM singer", "SELECT 3", "easy"),
    (
        "SELECT name FROM singer WHERE age > 35 ORDER BY age",
        "SELECT name FROM singer WHERE age > 35 ORDER BY age",
        "medium",
    ),
    # The average age is 41.3 here (only Joe is older) but 36 in the other version
    (ALL_AGE, "SELECT name FROM singer WHERE age > 40", "hard"),
    (
        "SELECT name FROM singer WHERE age > 35 AND name LIKE 'T%' ORDER BY age LIMIT 1",
        "SELEC name",
        "extra",
    ),
    # The gold returns no rows, and an empty prediction must still be wrong
    (ALL_AGE + " AND name LIKE 'T%' ORDER BY age", "", "extra"),
]


def make_singer_database(path: Path, extra_singer: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(1, "Joe", 52), (2, "Tim", 32), (3, "Ann", 40)] + (
        [(4, "Zed", 20)] if extra_singer else []
    )
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE singer (singer_id int PRIMARY KEY, name text, age int)")
    con.executemany("INSERT INTO singer VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()


@pytest.fixture
def scoring_spider_dir(tmp_path: Path) -> Path:
    root = tmp_path / "spider"
    root.mkdir()
    dev = [{"db_id": "concert", "question": "?", "query": gold} for gold, _, _ in QUESTIONS]
    (root / "dev.json").write_text(json.dumps(dev))
    (root / "tables.json").write_text(json.dumps(TABLES))
    make_singer_database(root / "database" / "concert" / "concert.sqlite", extra_singer=False)
    return root


@pytest.fixture
def suite_dir(tmp_path: Path) -> Path:
    root = tmp_path / "suite"
    make_singer_database(root / "concert" / "concert.sqlite", extra_singer=False)  # the original
    make_singer_database(root / "concert" / "concert_v2.sqlite", extra_singer=True)
    return root


@pytest.fixture
def predictions() -> list[Prediction]:
    return [
        Prediction(id=i, db_id="concert", predicted=predicted)
        for i, (_, predicted, _) in enumerate(QUESTIONS)
    ]


def make_test_split(root: Path, queries: list[str]) -> Path:
    """A Spider directory with only the test split: `test.json`, `test_tables.json`, `test_database/`."""
    make_singer_database(root / "test_database" / "concert" / "concert.sqlite", extra_singer=False)
    test = [{"db_id": "concert", "question": "?", "query": query} for query in queries]
    (root / "test.json").write_text(json.dumps(test))
    (root / "test_tables.json").write_text(spider_tables := json.dumps(TABLES))
    assert spider_tables
    return root


def fenced(gold: str) -> str:
    return (
        f"To answer the question, use this query:\n\n```sql\n{gold};\n```\n\nIt returns the rows."
    )
