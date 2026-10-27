"""Tests for text2sql.training.train, with stand-ins for the training libraries (no GPU or install needed)."""

import json
import types
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from conftest import TRAIN_CONFIG, FakeDataset, FakeTokenizer, make_sample


class FakeData:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def float(self) -> "FakeData":
        return FakeData("float32")


class FakeParam:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad
        self.data = FakeData("bfloat16")  # what TRL leaves on the adapter of a 4-bit model


class FakeModel:
    def __init__(self) -> None:
        self.params = [FakeParam(True), FakeParam(False), FakeParam(True)]

    def parameters(self) -> list[FakeParam]:
        return self.params


class FakeConfig:
    """Records the keyword arguments, for `LoraConfig` and `SFTConfig`."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeTrainer:
    instances: ClassVar[list["FakeTrainer"]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.model: Any = kwargs["model"]
        self.state = SimpleNamespace(log_history=[{"loss": 0.5, "step": 25}])
        self.dtypes_when_training: list[str] = []
        self.saved_to: str | None = None
        self.resumed: bool | None = None
        FakeTrainer.instances.append(self)

    def train(self, resume_from_checkpoint: bool = False) -> None:
        self.resumed = resume_from_checkpoint
        self.dtypes_when_training = [
            p.data.dtype for p in self.model.parameters() if p.requires_grad
        ]

    def save_model(self, path: str) -> None:
        self.saved_to = path
        Path(path).mkdir(parents=True, exist_ok=True)  # as the real trainer does


@pytest.fixture
def train_module(
    stub_module: Callable[..., types.ModuleType],
    fresh_import: Callable[[str], types.ModuleType],
) -> types.ModuleType:
    """`text2sql.training.train` imported against the stand-ins."""
    FakeTrainer.instances = []
    stub_module("datasets", Dataset=FakeDataset)
    stub_module("peft", LoraConfig=FakeConfig)
    stub_module("trl", SFTConfig=FakeConfig, SFTTrainer=FakeTrainer)
    stub_module("transformers", PreTrainedModel=FakeModel, PreTrainedTokenizerBase=FakeTokenizer)
    fresh_import("training.data")  # train imports it, and both are built on the stand-ins
    return fresh_import("training.train")


CONFIG = TRAIN_CONFIG
MAX_LENGTH = CONFIG["data"]["max_length"]


def test_train_passes_the_settings_to_the_trainer(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    model, tokenizer = FakeModel(), FakeTokenizer()

    train_module.train(model, tokenizer, [make_sample()], tmp_path / "adapter", CONFIG)

    (trainer,) = FakeTrainer.instances
    assert trainer.kwargs["model"] is model
    assert trainer.kwargs["processing_class"] is tokenizer
    assert len(trainer.kwargs["train_dataset"]) == 1
    assert trainer.kwargs["peft_config"].kwargs == CONFIG["lora"]
    assert trainer.kwargs["args"].kwargs == {
        "output_dir": str(tmp_path / "adapter"),
        "max_length": MAX_LENGTH,
        **CONFIG["training"],
    }


def test_train_casts_the_adapter_to_fp32_before_training(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    model = FakeModel()

    train_module.train(model, FakeTokenizer(), [make_sample()], tmp_path, CONFIG)

    (trainer,) = FakeTrainer.instances
    assert trainer.dtypes_when_training == ["float32", "float32"]  # the trainable parameters
    assert model.params[1].data.dtype == "bfloat16"  # the frozen one is left alone


def test_train_saves_the_adapter_and_the_log(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    train_module.train(FakeModel(), FakeTokenizer(), [make_sample()], tmp_path, CONFIG)

    (trainer,) = FakeTrainer.instances
    assert trainer.saved_to == str(tmp_path)
    assert json.loads((tmp_path / "training.log.json").read_text()) == [{"loss": 0.5, "step": 25}]


def test_train_returns_the_sample_counts(train_module: types.ModuleType, tmp_path: Path) -> None:
    samples = [make_sample(), make_sample(schema_words=MAX_LENGTH)]

    counts = train_module.train(FakeModel(), FakeTokenizer(), samples, tmp_path, CONFIG)

    assert counts == {"samples": 1, "dropped_too_long": 1}


def test_a_fresh_run_does_not_resume(train_module: types.ModuleType, tmp_path: Path) -> None:
    train_module.train(FakeModel(), FakeTokenizer(), [make_sample()], tmp_path, CONFIG)

    (trainer,) = FakeTrainer.instances
    assert trainer.resumed is False


def test_a_run_with_a_checkpoint_resumes_from_it(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "checkpoint-4").mkdir()

    train_module.train(FakeModel(), FakeTokenizer(), [make_sample()], tmp_path, CONFIG)

    (trainer,) = FakeTrainer.instances
    assert trainer.resumed is True


def test_checkpoints_are_removed_when_the_run_finishes_and_the_rest_is_kept(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "checkpoint-4").mkdir()
    (tmp_path / "checkpoint-4" / "optimizer.pt").write_text("state")
    (tmp_path / "adapter_model.safetensors").write_text("weights")

    train_module.train(FakeModel(), FakeTokenizer(), [make_sample()], tmp_path, CONFIG)

    assert not list(tmp_path.glob("checkpoint-*"))
    assert (tmp_path / "adapter_model.safetensors").read_text() == "weights"
    assert (tmp_path / "training.log.json").exists()
