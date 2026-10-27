"""Tests for `_generate_batch` in text2sql.generation, with a stand-in for the model."""

import types
from collections.abc import Callable
from typing import Any

import pytest
from conftest import PROJECT_ROOT

from text2sql.config import load_model_config
from text2sql.prompt import Message

CONFIG = load_model_config(PROJECT_ROOT / "configs" / "model.yaml")
PROMPT_IDS = [10, 11, 12]
CONVERSATIONS = [
    [Message(role="system", content="SCHEMA"), Message(role="user", content="FIRST")],
    [Message(role="system", content="SCHEMA"), Message(role="user", content="SECOND")],
]


def new_ids(row: int) -> list[int]:
    """The two tokens the stand-in model writes for a row."""
    return [100 + row, 200 + row]


class FakeTokens(list[Any]):
    """Mimics a 2D tensor of token ids, which `_generate_batch` slices as `output[:, n:]`."""

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

    def __init__(self) -> None:
        self.generate_kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: object) -> FakeTokens:
        self.generate_kwargs = kwargs
        rows = kwargs["input_ids"].shape[0]  # type: ignore[attr-defined]
        return FakeTokens([PROMPT_IDS + new_ids(row) for row in range(rows)])  # prompt, then reply


class FakeTokenizer:
    padding_side = "right"  # the default of the Qwen tokenizer

    def __init__(self) -> None:
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


@pytest.fixture
def generate_module(
    stub_module: Callable[..., types.ModuleType],
    fresh_import: Callable[[str], types.ModuleType],
) -> types.ModuleType:
    """`text2sql.generation` imported against the stand-ins."""
    stub_module("transformers", PreTrainedModel=FakeModel, PreTrainedTokenizerBase=FakeTokenizer)
    return fresh_import("generation")


def test_generate_batch_returns_the_new_tokens_of_each_conversation(
    generate_module: types.ModuleType,
) -> None:
    model, tokenizer = FakeModel(), FakeTokenizer()

    replies = generate_module._generate_batch(model, tokenizer, CONVERSATIONS, CONFIG)

    assert replies == ["100-200", "101-201"]  # the prompts are cut off, the order is kept
    assert tokenizer.decode_call == {
        "ids": [new_ids(0), new_ids(1)],
        "skip_special_tokens": True,
    }


def test_generate_batch_is_plain_greedy(generate_module: types.ModuleType) -> None:
    model, tokenizer = FakeModel(), FakeTokenizer()

    generate_module._generate_batch(model, tokenizer, CONVERSATIONS, CONFIG)

    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["repetition_penalty"] == 1.0
    # The longest gold query is 202 Qwen tokens (train_spider), counting the closing <|im_end|>
    assert model.generate_kwargs["max_new_tokens"] >= 202
    assert "padding_side" not in model.generate_kwargs  # it is a tokenizer setting
    assert model.generate_kwargs["input_ids"] is tokenizer.batch["input_ids"]
    assert model.generate_kwargs["attention_mask"] == "MASK"


def test_generate_batch_formats_every_conversation_and_tokenizes_them_together(
    generate_module: types.ModuleType,
) -> None:
    model, tokenizer = FakeModel(), FakeTokenizer()

    generate_module._generate_batch(model, tokenizer, CONVERSATIONS, CONFIG)

    assert tokenizer.templated == [
        {"messages": messages, "tokenize": False, "add_generation_prompt": True}
        for messages in CONVERSATIONS
    ]
    assert tokenizer.tokenize_call["texts"] == ["PROMPT FIRST", "PROMPT SECOND"]
    assert tokenizer.tokenize_call["add_special_tokens"] is False
    assert tokenizer.tokenize_call["padding"] is True
    assert tokenizer.tokenize_call["return_tensors"] == "pt"
    assert tokenizer.batch.device == "cuda:0"


def test_generate_batch_pads_on_the_left(generate_module: types.ModuleType) -> None:
    model, tokenizer = FakeModel(), FakeTokenizer()
    assert tokenizer.padding_side == "right"

    generate_module._generate_batch(model, tokenizer, CONVERSATIONS, CONFIG)

    assert tokenizer.tokenize_call["padding_side"] == "left"  # set before the prompts are padded


def test_generate_batch_works_for_a_single_conversation(generate_module: types.ModuleType) -> None:
    model, tokenizer = FakeModel(), FakeTokenizer()

    assert generate_module._generate_batch(model, tokenizer, CONVERSATIONS[:1], CONFIG) == [
        "100-200"
    ]
