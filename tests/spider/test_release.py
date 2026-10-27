"""Checks against the real Spider release; skipped when it is not downloaded."""

import pytest
from conftest import PROJECT_ROOT

from text2sql.spider.database import execution_error
from text2sql.spider.dataset import Split, load_examples
from text2sql.spider.samples import build_samples

SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"

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
    failing = [
        ex
        for ex in load_examples(SPIDER_DIR, split)
        if execution_error(SPIDER_DIR, ex["db_id"], ex["query"], split)
    ]

    assert failing == []
