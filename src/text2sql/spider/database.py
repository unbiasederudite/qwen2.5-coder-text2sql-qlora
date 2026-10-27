"""Spider's SQLite databases."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from text2sql.spider.dataset import Split, database_dir


@contextmanager
def _connect(spider_dir: Path, db_id: str, split: Split) -> Iterator[sqlite3.Connection]:
    """Opens one database read-only and closes it afterwards.

    Args:
        spider_dir (Path): Spider directory.
        db_id (str): Database name.
        split (Split): Split whose databases are wanted.

    Yields:
        sqlite3.Connection: Read-only connection.

    Raises:
        FileNotFoundError: If the database file is missing.
    """
    path = database_dir(spider_dir, split) / db_id / f"{db_id}.sqlite"
    if not path.exists():
        raise FileNotFoundError(path)
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as con:
        yield con


def load_schema_ddl(spider_dir: Path, db_id: str, split: Split) -> list[str]:
    """Reads the `CREATE TABLE` statements of one database.

    Args:
        spider_dir (Path): Spider directory.
        db_id (str): Database name.
        split (Split): Split whose databases are wanted.

    Returns:
        list[str]: One `CREATE TABLE` statement per table.

    Raises:
        FileNotFoundError: If the database file is missing.
    """
    with _connect(spider_dir, db_id, split) as con:
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY rowid"
        ).fetchall()
    return [f"{sql};" for (sql,) in rows]


def execution_error(spider_dir: Path, db_id: str, query: str, split: Split) -> str | None:
    """Runs a query on one database and returns the error it raises, if any.

    Args:
        spider_dir (Path): Spider directory.
        db_id (str): Database name.
        query (str): SQL to run.
        split (Split): Split whose databases are wanted.

    Returns:
        str | None: SQLite error message, or None if the query runs.

    Raises:
        FileNotFoundError: If the database file is missing.
    """
    with _connect(spider_dir, db_id, split) as con:
        try:
            con.execute(query).fetchone()
        except sqlite3.Error as e:
            return str(e)
    return None
