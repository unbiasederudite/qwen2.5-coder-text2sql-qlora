"""Text-to-SQL samples built from Spider."""

from pathlib import Path

from text2sql.spider.database import execution_error, load_schema_ddl
from text2sql.spider.dataset import Example, Split, load_examples


class Sample(Example):
    """One Spider example with its database schema."""

    schema: list[str]  # CREATE TABLE statements


def build_samples(spider_dir: Path, split: Split, clean: bool = False) -> list[Sample]:
    """Builds the samples of one split.

    Args:
        spider_dir (Path): Spider directory.
        split (Split): Split to build.
        clean (bool): Drop repeated questions and failing gold SQL.

    Returns:
        list[Sample]: Samples in file order.
    """
    ddl_by_db: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    samples: list[Sample] = []
    for ex in load_examples(spider_dir, split):
        db_id = ex["db_id"]
        if clean:
            key = (db_id, ex["question"])
            if key in seen or execution_error(spider_dir, db_id, ex["query"], split):
                continue
            seen.add(key)
        if db_id not in ddl_by_db:
            ddl_by_db[db_id] = load_schema_ddl(spider_dir, db_id, split)
        samples.append(Sample(**ex, schema=ddl_by_db[db_id]))
    return samples
