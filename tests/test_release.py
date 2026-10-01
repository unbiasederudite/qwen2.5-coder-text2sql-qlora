"""Checks against the real Spider release; skipped when it is not downloaded."""

from pathlib import Path

import pytest

from text2sql.data import build_samples
from text2sql.spider import Split, load_examples, query_error

SPIDER_DIR = Path(__file__).parent.parent / "data" / "spider_data"

pytestmark = pytest.mark.skipif(not SPIDER_DIR.exists(), reason="Spider is not downloaded")


@pytest.mark.parametrize(
    ("split", "size"),
    [("train_spider", 7000), ("train_others", 1659), ("dev", 1034), ("test", 2147)],
)
def test_split_sizes(split: Split, size: int) -> None:
    assert len(load_examples(SPIDER_DIR, split)) == size


def test_clean_train_spider_size() -> None:
    assert len(build_samples(SPIDER_DIR, "train_spider", clean=True)) == 6988


@pytest.mark.parametrize("split", ["dev", "test"])
def test_every_evaluation_query_runs(split: Split) -> None:
    test = split == "test"
    failing = [
        ex
        for ex in load_examples(SPIDER_DIR, split)
        if query_error(SPIDER_DIR, ex["db_id"], ex["query"], test=test)
    ]

    assert failing == []
