"""Tests for text2sql.config."""

from pathlib import Path
from typing import get_args

import pytest
import yaml
from conftest import PROJECT_ROOT, TRAIN_CONFIG

from text2sql import config
from text2sql.spider.dataset import Split

MODEL_CFG = {
    "supported_models": ["org/model"],
    "dtype": "float16",
    "quantization": {"load_in_4bit": True},
    "tokenizer": {"padding_side": "left"},
    "generation": {"max_new_tokens": 8, "do_sample": False},
}


def write_model_config(path: Path, **changes: object) -> Path:
    """Writes `MODEL_CFG` as a YAML file, with some settings replaced."""
    path.write_text(yaml.safe_dump({**MODEL_CFG, **changes}))
    return path


def test_load_model_config_reads_all_the_settings(tmp_path: Path) -> None:
    assert config.load_model_config(write_model_config(tmp_path / "c.yaml")) == MODEL_CFG


def test_load_model_config_rejects_an_empty_model_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supported_models must not be empty"):
        config.load_model_config(write_model_config(tmp_path / "c.yaml", supported_models=[]))


def test_load_model_config_rejects_an_unknown_padding_side(tmp_path: Path) -> None:
    path = write_model_config(tmp_path / "c.yaml", tokenizer={"padding_side": "middle"})

    with pytest.raises(ValueError, match="padding_side must be one of"):
        config.load_model_config(path)


def test_load_model_config_rejects_no_new_tokens(tmp_path: Path) -> None:
    path = write_model_config(tmp_path / "c.yaml", generation={"max_new_tokens": 0})

    with pytest.raises(ValueError, match="max_new_tokens must be at least 1, not 0"):
        config.load_model_config(path)


def test_the_shipped_model_config_loads_with_the_values_yaml_could_misread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)  # the default path is from the project root

    loaded = config.load_model_config()

    assert "Qwen/Qwen2.5-Coder-1.5B-Instruct" in loaded["supported_models"]
    assert loaded["quantization"]["load_in_4bit"] is True
    assert loaded["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert loaded["quantization"]["bnb_4bit_compute_dtype"] == "float16"
    assert loaded["generation"]["do_sample"] is False
    assert loaded["generation"]["repetition_penalty"] == 1.0
    # The longest gold query is 202 Qwen tokens (train_spider), counting the closing <|im_end|>
    assert loaded["generation"]["max_new_tokens"] >= 202
    assert "padding_side" not in loaded["generation"]  # it is a tokenizer setting
    assert loaded["tokenizer"]["padding_side"] == "left"


def write_train_config(path: Path, **training: object) -> Path:
    """Writes `TRAIN_CONFIG` as a YAML file, with some training settings replaced."""
    config = {**TRAIN_CONFIG, "training": {**TRAIN_CONFIG["training"], **training}}
    path.write_text(yaml.safe_dump(config))
    return path


def test_load_train_config_reads_the_splits_and_the_settings(tmp_path: Path) -> None:
    assert config.load_train_config(write_train_config(tmp_path / "c.yaml")) == TRAIN_CONFIG


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("num_train_epochs", 0, "num_train_epochs must be positive, not 0"),
        ("per_device_train_batch_size", 0, "per_device_train_batch_size must be at least 1, not 0"),
        ("learning_rate", 0.0, "learning_rate must be positive, not 0.0"),
    ],
)
def test_load_train_config_rejects_values_out_of_range(
    tmp_path: Path, setting: str, value: float, message: str
) -> None:
    path = write_train_config(tmp_path / "c.yaml", **{setting: value})

    with pytest.raises(ValueError, match=message):
        config.load_train_config(path)


@pytest.mark.parametrize("splits", [[], ["dev"], ["train_spider", "test"]])
def test_load_train_config_rejects_splits_that_are_not_for_training(
    tmp_path: Path, splits: list[str]
) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump({**TRAIN_CONFIG, "data": {**TRAIN_CONFIG["data"], "splits": splits}})
    )

    with pytest.raises(ValueError, match="splits must be a non-empty subset"):
        config.load_train_config(path)


def test_load_train_config_rejects_a_max_length_below_one(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump({**TRAIN_CONFIG, "data": {**TRAIN_CONFIG["data"], "max_length": 0}})
    )

    with pytest.raises(ValueError, match="max_length must be at least 1, not 0"):
        config.load_train_config(path)


def test_the_shipped_train_config_loads_with_the_values_yaml_could_misread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)  # the default path is from the project root

    loaded = config.load_train_config()

    assert set(loaded["data"]["splits"]) <= set(get_args(Split))
    assert loaded["data"]["clean"] is True
    assert loaded["data"]["max_length"] == 2048
    assert "max_length" not in loaded["training"]  # it is a data setting
    assert loaded["training"]["save_strategy"] == "epoch"
    assert loaded["training"]["report_to"] == "none"
    assert loaded["lora"]["bias"] == "none"
    assert isinstance(loaded["training"]["learning_rate"], float)  # 2e-4 would be a string
