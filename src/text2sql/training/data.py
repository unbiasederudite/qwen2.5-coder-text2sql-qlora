"""The training dataset: Spider samples as prompt-completion rows that fit the token limit."""

from pathlib import Path
from typing import TYPE_CHECKING

from text2sql.config import DataConfig
from text2sql.prompt import Message, build_completion, build_prompt
from text2sql.spider.samples import Sample, build_samples

if TYPE_CHECKING:  # only `build_dataset` needs the training libraries, the rest runs anywhere
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase


def build_training_samples(spider_dir: Path, data: DataConfig) -> list[Sample]:
    """Builds the samples of the training splits.

    Args:
        spider_dir (Path): Spider directory.
        data (DataConfig): Data settings from `load_train_config`.

    Returns:
        list[Sample]: Samples of every split, in the order of the splits.
    """
    return [
        sample
        for split in data["splits"]
        for sample in build_samples(spider_dir, split, clean=data["clean"])
    ]


def _token_count(tokenizer: "PreTrainedTokenizerBase", messages: list[Message]) -> int:
    """Counts the tokens of a conversation.

    Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer from `load_model`.
        messages (list[Message]): Conversation messages.

    Returns:
        int: Number of tokens.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_dataset(
    samples: list[Sample], tokenizer: "PreTrainedTokenizerBase", max_length: int
) -> tuple["Dataset", int]:
    """Builds the prompt-completion dataset.

    Args:
        samples (list[Sample]): Samples to train on.
        tokenizer (PreTrainedTokenizerBase): Tokenizer from `load_model`.
        max_length (int): Longest sample to keep, in tokens.

    Returns:
        tuple[Dataset, int]: Dataset, and the number of samples dropped for length.
    """
    from datasets import Dataset  # installed only on Colab

    rows = [{"prompt": build_prompt(s), "completion": build_completion(s)} for s in samples]
    kept = [
        row
        for row in rows
        if _token_count(tokenizer, row["prompt"] + row["completion"]) <= max_length
    ]
    return Dataset.from_list(kept), len(rows) - len(kept)
