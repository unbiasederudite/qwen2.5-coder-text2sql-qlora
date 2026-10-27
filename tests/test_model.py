"""Tests for text2sql.model, with stand-ins for transformers (no GPU or install needed)."""

import types
from collections.abc import Callable
from typing import Any

import pytest
from conftest import PROJECT_ROOT

from text2sql.config import load_model_config

CONFIG = load_model_config(PROJECT_ROOT / "configs" / "model.yaml")


class FakeModel:
    def __init__(self, name: str, load_kwargs: dict[str, Any]) -> None:
        self.name = name
        self.load_kwargs = load_kwargs
        self.adapters: list[str] = []

    def load_adapter(self, adapter: str) -> None:
        self.adapters.append(adapter)


class FakeTokenizer:
    def __init__(self, name: str) -> None:
        self.name = name


class BitsAndBytesConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(name: str, **kwargs: object) -> FakeModel:
        return FakeModel(name, kwargs)


class AutoTokenizer:
    @staticmethod
    def from_pretrained(name: str) -> FakeTokenizer:
        return FakeTokenizer(name)


@pytest.fixture
def model_module(
    stub_module: Callable[..., types.ModuleType],
    fresh_import: Callable[[str], types.ModuleType],
) -> types.ModuleType:
    """`text2sql.model` imported against the stand-ins."""
    stub_module(
        "transformers",
        BitsAndBytesConfig=BitsAndBytesConfig,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        PreTrainedModel=FakeModel,
        PreTrainedTokenizerBase=FakeTokenizer,
    )
    return fresh_import("model")


def test_load_model_passes_the_config_to_the_library(model_module: types.ModuleType) -> None:
    model, _ = model_module.load_model("some/model", CONFIG)

    assert model.name == "some/model"
    assert model.load_kwargs["quantization_config"].kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
    }
    assert model.load_kwargs["dtype"] == "float16"


def test_load_model_leaves_device_placement_to_the_library(model_module: types.ModuleType) -> None:
    model, _ = model_module.load_model("some/model", CONFIG)

    assert "device_map" not in model.load_kwargs


def test_load_model_returns_the_tokenizer_of_the_same_model(model_module: types.ModuleType) -> None:
    _, tokenizer = model_module.load_model("some/model", CONFIG)

    assert tokenizer.name == "some/model"


def test_load_model_puts_the_adapter_on_the_model(model_module: types.ModuleType) -> None:
    model, tokenizer = model_module.load_model("some/model", CONFIG, "some/adapter")

    assert model.adapters == ["some/adapter"]
    assert model.name == "some/model"  # the base model is still the one that is loaded
    assert tokenizer.name == "some/model"


def test_load_model_without_an_adapter_loads_none(model_module: types.ModuleType) -> None:
    model, _ = model_module.load_model("some/model", CONFIG)

    assert model.adapters == []


def test_model_slug_drops_the_organization_and_lowers_the_case(
    model_module: types.ModuleType,
) -> None:
    slug = model_module.model_slug("Qwen/Qwen2.5-Coder-1.5B-Instruct")

    assert slug == "qwen2.5-coder-1.5b-instruct"
