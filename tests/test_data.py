from pathlib import Path

from conftest import CONCERT_DDL, SHOP_DDL, TRAIN

from text2sql.data import build_samples


def test_samples_get_the_schema_of_their_own_database(spider_dir: Path) -> None:
    samples = build_samples(spider_dir, "train_spider", clean=False)

    expected = {"concert": CONCERT_DDL, "shop": SHOP_DDL}
    for sample in samples:
        assert sample["schema"] == [f"{s};" for s in expected[sample["db_id"]]]


def test_samples_extend_examples_in_file_order(spider_dir: Path) -> None:
    samples = build_samples(spider_dir, "train_spider", clean=False)

    assert [{k: s[k] for k in ("db_id", "question", "query")} for s in samples] == TRAIN


def test_clean_false_keeps_every_example(spider_dir: Path) -> None:
    assert len(build_samples(spider_dir, "train_spider", clean=False)) == len(TRAIN)


def test_clean_drops_repeated_questions_and_failing_sql(spider_dir: Path) -> None:
    samples = build_samples(spider_dir, "train_spider", clean=True)

    assert [s["question"] for s in samples] == [
        "How many singers are there?",
        "List the shop names.",
    ]


def test_clean_is_off_by_default(spider_dir: Path) -> None:
    assert build_samples(spider_dir, "train_spider") == build_samples(
        spider_dir, "train_spider", clean=False
    )


def test_test_split_reads_test_databases(spider_dir: Path) -> None:
    (sample,) = build_samples(spider_dir, "test")

    assert sample["db_id"] == "museum"
