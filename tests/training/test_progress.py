"""Tests for text2sql.training.progress, with a stand-in for the progress bar of transformers."""

import types
from collections.abc import Callable

import pytest


class FakeBar:
    def __init__(self) -> None:
        self.postfix: dict[str, str] = {}

    def set_postfix(self, **kwargs: str) -> None:
        self.postfix = kwargs


class FakeProgressCallback:
    training_bar: FakeBar | None = None


@pytest.fixture
def progress_module(
    stub_module: Callable[..., types.ModuleType],
    fresh_import: Callable[[str], types.ModuleType],
) -> types.ModuleType:
    """`text2sql.training.progress` imported against the stand-in."""
    stub_module("transformers.trainer_callback", ProgressCallback=FakeProgressCallback)
    stub_module("transformers")
    return fresh_import("training.progress")


LOGS = {"loss": 0.31654}


def test_progress_callback_shows_metrics_after_the_bar(progress_module: types.ModuleType) -> None:
    callback = progress_module.MetricsProgressCallback()
    callback.training_bar = FakeBar()
    callback.on_log(None, None, None, logs=LOGS)
    assert callback.training_bar.postfix == {"loss": "0.3165"}


def test_progress_callback_ignores_summary_and_missing_bar(
    progress_module: types.ModuleType,
) -> None:
    callback = progress_module.MetricsProgressCallback()
    callback.on_log(None, None, None, logs=LOGS)  # no bar yet
    callback.training_bar = FakeBar()
    callback.on_log(None, None, None, logs={"train_runtime": 1.0})
    assert callback.training_bar.postfix == {}
