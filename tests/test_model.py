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
CONVERSATIONS = [
    [Message(role="system", content="SCHEMA"), Message(role="user", content="FIRST")],
    [Message(role="system", content="SCHEMA"), Message(role="user", content="SECOND")],
]


def new_ids(row: int) -> list[int]:
    """The two tokens the stand-in model writes for a row."""
    return [100 + row, 200 + row]


class FakeTokens(list[Any]):
    """Mimics a 2D tensor of token ids, which `generate_batch` slices as `output[:, n:]`."""

    def __getitem__(self, key: object) -> object:
        if isinstance(key, tuple):
            rows, columns = key
            return FakeTokens([row[columns] for row in super().__getitem__(rows)])
        return super().__getitem__(key)


class FakeIds:
    def __init__(self, rows: int) -> None:
        self.shape = (rows, len(PROMPT_IDS))


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
        rows = kwargs["input_ids"].shape[0]  # type: ignore[attr-defined]
        return FakeTokens([PROMPT_IDS + new_ids(row) for row in range(rows)])  # prompt, then reply


class FakeTokenizer:
    padding_side = "right"  # the default of the Qwen tokenizer

    def __init__(self, name: str) -> None:
        self.name = name
        self.templated: list[dict[str, Any]] = []
        self.tokenize_call: dict[str, Any] = {}
        self.decode_call: dict[str, Any] = {}
        self.batch = FakeBatch()

    def apply_chat_template(
        self, messages: list[Message], tokenize: bool, add_generation_prompt: bool
    ) -> str:
        self.templated.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return f"PROMPT {messages[-1]['content']}"

    def __call__(
        self, texts: list[str], add_special_tokens: bool, padding: bool, return_tensors: str
    ) -> FakeBatch:
        self.tokenize_call = {
            "texts": texts,
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "padding_side": self.padding_side,  # what it is when the prompts are tokenized
            "return_tensors": return_tensors,
        }
        self.batch = FakeBatch(input_ids=FakeIds(len(texts)), attention_mask="MASK")
        return self.batch

    def batch_decode(self, ids: list[list[int]], skip_special_tokens: bool) -> list[str]:
        self.decode_call = {
            "ids": [list(row) for row in ids],
            "skip_special_tokens": skip_special_tokens,
        }
        return ["-".join(map(str, row)) for row in ids]


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


def test_generate_batch_returns_the_new_tokens_of_each_conversation(
    model_module: types.ModuleType,
) -> None:
    model, tokenizer = model_module.load_model("some/model")

    replies = model_module.generate_batch(model, tokenizer, CONVERSATIONS)

    assert replies == ["100-200", "101-201"]  # the prompts are cut off, the order is kept
    assert tokenizer.decode_call == {
        "ids": [new_ids(0), new_ids(1)],
        "skip_special_tokens": True,
    }


def test_generate_batch_is_plain_greedy(model_module: types.ModuleType) -> None:
    model, tokenizer = model_module.load_model("some/model")

    model_module.generate_batch(model, tokenizer, CONVERSATIONS)

    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["repetition_penalty"] == 1.0
    # the longest gold query is 202 Qwen tokens (train_spider), counting the closing <|im_end|>
    assert model.generate_kwargs["max_new_tokens"] >= 202
    assert "padding_side" not in model.generate_kwargs  # it is a tokenizer setting
    assert model.generate_kwargs["input_ids"] is tokenizer.batch["input_ids"]
    assert model.generate_kwargs["attention_mask"] == "MASK"


def test_generate_batch_formats_every_conversation_and_tokenizes_them_together(
    model_module: types.ModuleType,
) -> None:
    model, tokenizer = model_module.load_model("some/model")

    model_module.generate_batch(model, tokenizer, CONVERSATIONS)

    assert tokenizer.templated == [
        {"messages": messages, "tokenize": False, "add_generation_prompt": True}
        for messages in CONVERSATIONS
    ]
    assert tokenizer.tokenize_call["texts"] == ["PROMPT FIRST", "PROMPT SECOND"]
    assert tokenizer.tokenize_call["add_special_tokens"] is False
    assert tokenizer.tokenize_call["padding"] is True
    assert tokenizer.tokenize_call["return_tensors"] == "pt"
    assert tokenizer.batch.device == "cuda:0"


def test_generate_batch_pads_on_the_left(model_module: types.ModuleType) -> None:
    model, tokenizer = model_module.load_model("some/model")
    assert tokenizer.padding_side == "right"

    model_module.generate_batch(model, tokenizer, CONVERSATIONS)

    assert tokenizer.tokenize_call["padding_side"] == "left"  # set before the prompts are padded


def test_generate_batch_works_for_a_single_conversation(model_module: types.ModuleType) -> None:
    model, tokenizer = model_module.load_model("some/model")

    assert model_module.generate_batch(model, tokenizer, CONVERSATIONS[:1]) == ["100-200"]


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
