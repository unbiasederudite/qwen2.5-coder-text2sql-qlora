"""The model and training config files, and their checks."""

from pathlib import Path
from typing import Any, TypedDict

import yaml

from text2sql.spider.dataset import TRAIN_SPLITS, Split

# Default config files, from the project root
_MODEL_CONFIG_PATH = Path("configs/model.yaml")
_TRAIN_CONFIG_PATH = Path("configs/train.yaml")
_PADDING_SIDES = ("left", "right")


class TokenizerConfig(TypedDict):
    """The tokenizer settings of a model config file."""

    padding_side: str  # one of `_PADDING_SIDES`


class ModelConfig(TypedDict):
    """One model config file."""

    supported_models: list[str]  # Hugging Face model names that are accepted
    dtype: str  # type of the weights that are not quantized
    quantization: dict[str, Any]  # arguments of `transformers.BitsAndBytesConfig`
    tokenizer: TokenizerConfig  # settings set on the tokenizer
    generation: dict[str, Any]  # arguments of `model.generate`


def load_model_config(path: Path = _MODEL_CONFIG_PATH) -> ModelConfig:
    """Reads a model config file and checks its values.

    Args:
        path (Path): YAML config file, `configs/model.yaml` by default.

    Returns:
        ModelConfig: Supported models, dtype, quantization, tokenizer and generation settings.

    Raises:
        ValueError: If a setting is not allowed or out of range.
    """
    config: ModelConfig = yaml.safe_load(path.read_text())
    if not config["supported_models"]:
        raise ValueError("supported_models must not be empty")
    padding_side = config["tokenizer"]["padding_side"]
    if padding_side not in _PADDING_SIDES:
        raise ValueError(f"padding_side must be one of {_PADDING_SIDES}, not {padding_side!r}")
    generation = config["generation"]
    if generation["max_new_tokens"] < 1:
        raise ValueError(f"max_new_tokens must be at least 1, not {generation['max_new_tokens']}")
    return config


class DataConfig(TypedDict):
    """The training samples of a training config file."""

    splits: list[Split]  # Spider training splits to train on, from `TRAIN_SPLITS`
    clean: bool  # drop repeated questions and gold SQL that fails to run
    max_length: int  # longest sample to keep, in tokens


class TrainConfig(TypedDict):
    """One training config file."""

    data: DataConfig  # which samples to train on
    lora: dict[str, Any]  # arguments of `peft.LoraConfig`
    training: dict[str, Any]  # arguments of `trl.SFTConfig`


def load_train_config(path: Path = _TRAIN_CONFIG_PATH) -> TrainConfig:
    """Reads a training config file and checks its values.

    Args:
        path (Path): YAML config file, `configs/train.yaml` by default.

    Returns:
        TrainConfig: Data, LoRA settings and training settings.

    Raises:
        ValueError: If a setting is not allowed or out of range.
    """
    config: TrainConfig = yaml.safe_load(path.read_text())
    data = config["data"]
    if not data["splits"] or not set(data["splits"]) <= set(TRAIN_SPLITS):
        raise ValueError(
            f"splits must be a non-empty subset of {TRAIN_SPLITS}, not {data['splits']}"
        )
    if data["max_length"] < 1:
        raise ValueError(f"max_length must be at least 1, not {data['max_length']}")
    training = config["training"]
    if training["num_train_epochs"] <= 0:
        raise ValueError(f"num_train_epochs must be positive, not {training['num_train_epochs']}")
    if training["per_device_train_batch_size"] < 1:
        raise ValueError(
            "per_device_train_batch_size must be at least 1, "
            f"not {training['per_device_train_batch_size']}"
        )
    if training["learning_rate"] <= 0:
        raise ValueError(f"learning_rate must be positive, not {training['learning_rate']}")
    return config
