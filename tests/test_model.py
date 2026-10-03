"""Tests for text2sql.model, with stand-ins for torch and transformers (no GPU or install needed)."""

import importlib
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

import text2sql
from text2sql.prompt import Message

PROMPT_IDS = [10, 11, 12]
NEW_IDS = [7, 8]
MESSAGES = [
    Message(role="system", content="SCHEMA"),
    Message(role="user", content="QUESTION"),
]


class FakeTokens(list[Any]):
    """Mimics a 2D tensor of token ids, which `generate` slices as `output[0, n:]`."""

    def __getitem__(self, key: object) -> object:
        if isinstance(key, tuple):
            row, rest = key
            return super().__getitem__(row)[rest]
        return super().__getitem__(key)


class FakeIds:
    shape = (1, len(PROMPT_IDS))


class FakeBatch(dict[str, Any]):
    device: str | None = None

    def to(self, device: str) -> "FakeBatch":
        self.device = device
        return self


class FakeModel:
    device = "cuda:0"

    def __init__(self, name: str, load_kwargs: dict[str, Any]) -> None:
        self.name = name
        self.load_kwargs = load_kwargs
        self.generate_kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: object) -> FakeTokens:
        self.generate_kwargs = kwargs
        return FakeTokens([PROMPT_IDS + NEW_IDS])  # the prompt, then the new tokens


class FakeTokenizer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.template_call: dict[str, Any] = {}
        self.tokenize_call: dict[str, Any] = {}
        self.decode_call: dict[str, Any] = {}
        self.batch = FakeBatch()

    def apply_chat_template(
        self, messages: list[Message], tokenize: bool, add_generation_prompt: bool
    ) -> str:
        self.template_call = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        return "TEMPLATED PROMPT"

    def __call__(self, text: str, add_special_tokens: bool, return_tensors: str) -> FakeBatch:
        self.tokenize_call = {
            "text": text,
            "add_special_tokens": add_special_tokens,
            "return_tensors": return_tensors,
        }
        self.batch = FakeBatch(input_ids=FakeIds(), attention_mask="MASK")
        return self.batch

    def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
        self.decode_call = {"ids": list(ids), "skip_special_tokens": skip_special_tokens}
        return "DECODED"


def fake_transformers() -> types.ModuleType:
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

    module = types.ModuleType("transformers")
    module.__dict__.update(
        BitsAndBytesConfig=BitsAndBytesConfig,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        PreTrainedModel=FakeModel,
        PreTrainedTokenizerBase=FakeTokenizer,
    )
    return module


@pytest.fixture
def model_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """`text2sql.model` imported against the stand-ins, and removed again afterwards."""
    torch = types.ModuleType("torch")
    torch.__dict__["float16"] = "FLOAT16"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers())
    monkeypatch.delitem(sys.modules, "text2sql.model", raising=False)
    monkeypatch.delattr(text2sql, "model", raising=False)
    yield importlib.import_module("text2sql.model")
    sys.modules.pop("text2sql.model", None)  # it was built on the stand-ins, so do not leak it
    text2sql.__dict__.pop("model", None)


def test_generate_returns_only_the_new_tokens(model_module: types.ModuleType) -> None:
    model, tokenizer = model_module.load_model("some/model")

    reply = model_module.generate(model, tokenizer, MESSAGES)

    assert reply == "DECODED"
    assert tokenizer.decode_call == {"ids": NEW_IDS, "skip_special_tokens": True}


def test_generate_is_plain_greedy(model_module: types.ModuleType) -> None:
    model, tokenizer = model_module.load_model("some/model")

    model_module.generate(model, tokenizer, MESSAGES)

    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["repetition_penalty"] == 1.0
    # the longest gold query is 202 Qwen tokens (train_spider), counting the closing <|im_end|>
    assert model.generate_kwargs["max_new_tokens"] >= 202
    assert model.generate_kwargs["input_ids"] is tokenizer.batch["input_ids"]
    assert model.generate_kwargs["attention_mask"] == "MASK"


def test_generate_passes_the_token_limit_through(model_module: types.ModuleType) -> None:
    model, tokenizer = model_module.load_model("some/model")

    model_module.generate(model, tokenizer, MESSAGES, max_new_tokens=256)

    assert model.generate_kwargs["max_new_tokens"] == 256


def test_generate_formats_the_prompt_with_the_chat_template(model_module: types.ModuleType) -> None:
    model, tokenizer = model_module.load_model("some/model")

    model_module.generate(model, tokenizer, MESSAGES)

    assert tokenizer.template_call == {
        "messages": MESSAGES,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    assert tokenizer.tokenize_call == {
        "text": "TEMPLATED PROMPT",
        "add_special_tokens": False,
        "return_tensors": "pt",
    }
    assert tokenizer.batch.device == "cuda:0"


def test_load_model_uses_4bit_nf4_with_double_quantization(model_module: types.ModuleType) -> None:
    model, _ = model_module.load_model("some/model")

    assert model.name == "some/model"
    assert model.load_kwargs["quantization_config"].kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "FLOAT16",
    }
    assert model.load_kwargs["dtype"] == "FLOAT16"


def test_load_model_leaves_device_placement_to_the_library(model_module: types.ModuleType) -> None:
    model, _ = model_module.load_model("some/model")

    assert "device_map" not in model.load_kwargs


def test_load_model_returns_the_tokenizer_of_the_same_model(model_module: types.ModuleType) -> None:
    _, tokenizer = model_module.load_model("some/model")

    assert tokenizer.name == "some/model"
