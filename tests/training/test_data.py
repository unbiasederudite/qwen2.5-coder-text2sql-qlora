"""Tests for text2sql.training.data, with stand-ins for the dataset libraries (no install needed)."""

import types
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import TRAIN_CONFIG, FakeDataset, FakeTokenizer, make_sample

from text2sql.prompt import build_completion, build_prompt
from text2sql.training.data import build_training_samples

DATA = TRAIN_CONFIG["data"]
MAX_LENGTH = DATA["max_length"]


@pytest.fixture
def data_module(
    stub_module: Callable[..., types.ModuleType],
    fresh_import: Callable[[str], types.ModuleType],
) -> types.ModuleType:
    """`text2sql.training.data` imported against the stand-ins, for `build_dataset`."""
    stub_module("datasets", Dataset=FakeDataset)
    stub_module("transformers", PreTrainedTokenizerBase=FakeTokenizer)
    return fresh_import("training.data")


def test_build_training_samples_cleans_the_samples_of_every_split(spider_dir: Path) -> None:
    samples = build_training_samples(spider_dir, DATA)

    assert [s["question"] for s in samples] == [
        "How many singers are there?",
        "List the shop names.",
    ]


def test_build_training_samples_keeps_the_repeats_unless_asked_to_clean(spider_dir: Path) -> None:
    samples = build_training_samples(spider_dir, {**DATA, "clean": False})

    assert len(samples) == 4


def test_token_count_counts_the_tokens_of_the_chat_template(
    data_module: types.ModuleType,
) -> None:
    messages = [{"role": "user", "content": "one two"}, {"role": "assistant", "content": "three"}]

    assert data_module._token_count(FakeTokenizer(), messages) == 3


def test_build_dataset_pairs_each_prompt_with_its_completion(
    data_module: types.ModuleType,
) -> None:
    samples = [make_sample(), make_sample(query="SELECT 2")]

    dataset, dropped = data_module.build_dataset(samples, FakeTokenizer(), MAX_LENGTH)

    assert dropped == 0
    assert dataset.rows == [
        {"prompt": build_prompt(s), "completion": build_completion(s)} for s in samples
    ]


def test_build_dataset_drops_the_samples_longer_than_the_limit(
    data_module: types.ModuleType,
) -> None:
    fits = make_sample(schema_words=MAX_LENGTH - 5)  # the question and the query add a few tokens
    too_long = make_sample(schema_words=MAX_LENGTH)

    dataset, dropped = data_module.build_dataset([fits, too_long], FakeTokenizer(), MAX_LENGTH)

    assert dropped == 1
    assert [row["prompt"][0]["content"] for row in dataset.rows] == [fits["schema"][0]]
