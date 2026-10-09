"""Tests for text2sql.train, with stand-ins for the training libraries (no GPU or install needed)."""

import json
import types
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, get_args

import pytest
import yaml

from text2sql.data import Sample
from text2sql.prompt import build_completion, build_prompt
from text2sql.spider import Split


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


class FakeTokenizer:
    """Counts a word as a token."""

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool) -> str:
        assert tokenize is False
        return " ".join(message["content"] for message in messages)

    def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[str]]:
        assert add_special_tokens is False
        return {"input_ids": text.split()}


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    @classmethod
    def from_list(cls, rows: list[dict[str, Any]]) -> "FakeDataset":
        return cls(rows)

    def __len__(self) -> int:
        return len(self.rows)


class FakeConfig:
    """Records the keyword arguments, for `LoraConfig` and `SFTConfig`."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeBar:
    def __init__(self) -> None:
        self.postfix: dict[str, str] = {}

    def set_postfix(self, **kwargs: str) -> None:
        self.postfix = kwargs


class FakeProgressCallback:
    training_bar: FakeBar | None = None


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
    """`text2sql.train` imported against the stand-ins."""
    FakeTrainer.instances = []
    stub_module("datasets", Dataset=FakeDataset)
    stub_module("peft", LoraConfig=FakeConfig)
    stub_module("trl", SFTConfig=FakeConfig, SFTTrainer=FakeTrainer)
    stub_module("transformers", PreTrainedModel=FakeModel, PreTrainedTokenizerBase=FakeTokenizer)
    stub_module("transformers.trainer_callback", ProgressCallback=FakeProgressCallback)
    return fresh_import("train")


CONFIG = {
    "splits": ["train_spider"],
    "lora": {"r": 16, "lora_alpha": 32},
    "training": {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 4,
        "learning_rate": 2e-4,
        "max_length": 50,  # the stand-in tokenizer counts a word as a token
        "seed": 7,
    },
}
MAX_LENGTH = CONFIG["training"]["max_length"]


def sample(schema_words: int = 1, query: str = "SELECT 1") -> Sample:
    return Sample(
        db_id="db",
        question="how many?",
        query=query,
        schema=[" ".join(["CREATE"] * schema_words)],
    )


def test_log_path_is_inside_the_adapter_directory(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    assert train_module.log_path(tmp_path) == tmp_path / "training.log.json"


def test_token_count_counts_the_tokens_of_the_chat_template(
    train_module: types.ModuleType,
) -> None:
    messages = [{"role": "user", "content": "one two"}, {"role": "assistant", "content": "three"}]

    assert train_module.token_count(FakeTokenizer(), messages) == 3


def test_build_dataset_pairs_each_prompt_with_its_completion(
    train_module: types.ModuleType,
) -> None:
    samples = [sample(), sample(query="SELECT 2")]

    dataset, dropped = train_module.build_dataset(samples, FakeTokenizer(), MAX_LENGTH)

    assert dropped == 0
    assert dataset.rows == [
        {"prompt": build_prompt(s), "completion": build_completion(s)} for s in samples
    ]


def test_build_dataset_drops_the_samples_longer_than_the_limit(
    train_module: types.ModuleType,
) -> None:
    fits = sample(schema_words=MAX_LENGTH - 5)  # the question and the query add a few tokens
    too_long = sample(schema_words=MAX_LENGTH)

    dataset, dropped = train_module.build_dataset([fits, too_long], FakeTokenizer(), MAX_LENGTH)

    assert dropped == 1
    assert [row["prompt"][0]["content"] for row in dataset.rows] == [fits["schema"][0]]


def test_train_passes_the_settings_to_the_trainer(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    model, tokenizer = FakeModel(), FakeTokenizer()

    train_module.train(model, tokenizer, [sample()], tmp_path / "adapter", CONFIG)

    (trainer,) = FakeTrainer.instances
    assert trainer.kwargs["model"] is model
    assert trainer.kwargs["processing_class"] is tokenizer
    assert len(trainer.kwargs["train_dataset"]) == 1
    assert trainer.kwargs["peft_config"].kwargs == CONFIG["lora"]
    assert trainer.kwargs["args"].kwargs == {
        "output_dir": str(tmp_path / "adapter"),
        **CONFIG["training"],
    }


def test_train_casts_the_adapter_to_fp32_before_training(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    model = FakeModel()

    train_module.train(model, FakeTokenizer(), [sample()], tmp_path, CONFIG)

    (trainer,) = FakeTrainer.instances
    assert trainer.dtypes_when_training == ["float32", "float32"]  # the trainable parameters
    assert model.params[1].data.dtype == "bfloat16"  # the frozen one is left alone


def test_train_saves_the_adapter_and_the_log(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    train_module.train(FakeModel(), FakeTokenizer(), [sample()], tmp_path, CONFIG)

    (trainer,) = FakeTrainer.instances
    assert trainer.saved_to == str(tmp_path)
    assert json.loads((tmp_path / "training.log.json").read_text()) == [{"loss": 0.5, "step": 25}]


def test_train_returns_the_sample_counts(train_module: types.ModuleType, tmp_path: Path) -> None:
    samples = [sample(), sample(schema_words=MAX_LENGTH)]

    counts = train_module.train(FakeModel(), FakeTokenizer(), samples, tmp_path, CONFIG)

    assert counts == {"samples": 1, "dropped_too_long": 1}


def write_config(path: Path, **training: object) -> Path:
    """Writes `CONFIG` as a YAML file, with some training settings replaced."""
    config = {**CONFIG, "training": {**CONFIG["training"], **training}}
    path.write_text(yaml.safe_dump(config))
    return path


def test_load_config_reads_the_splits_and_the_settings(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    assert train_module.load_config(write_config(tmp_path / "c.yaml")) == CONFIG


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("num_train_epochs", 0, "num_train_epochs must be positive, not 0"),
        ("per_device_train_batch_size", 0, "per_device_train_batch_size must be at least 1, not 0"),
        ("learning_rate", 0.0, "learning_rate must be positive, not 0.0"),
    ],
)
def test_load_config_rejects_values_out_of_range(
    train_module: types.ModuleType, tmp_path: Path, setting: str, value: float, message: str
) -> None:
    path = write_config(tmp_path / "c.yaml", **{setting: value})

    with pytest.raises(ValueError, match=message):
        train_module.load_config(path)


def test_the_shipped_config_loads_with_the_values_yaml_could_misread(
    train_module: types.ModuleType,
) -> None:
    config = train_module.load_config(Path(__file__).parent.parent / "configs" / "qlora.yaml")

    assert set(config["splits"]) <= set(get_args(Split))
    assert config["training"]["save_strategy"] == "epoch"
    assert config["training"]["report_to"] == "none"
    assert config["lora"]["bias"] == "none"
    assert isinstance(config["training"]["learning_rate"], float)  # 2e-4 would be a string


def test_has_checkpoint_is_true_only_with_a_checkpoint_in_the_directory(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    assert train_module.has_checkpoint(tmp_path) is False
    (tmp_path / "adapter_config.json").write_text("{}")
    assert train_module.has_checkpoint(tmp_path) is False
    (tmp_path / "checkpoint-4").mkdir()
    assert train_module.has_checkpoint(tmp_path) is True


def test_a_run_is_finished_only_with_the_adapter_and_no_checkpoint(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    assert train_module.is_finished(tmp_path) is False  # nothing there
    (tmp_path / "checkpoint-4").mkdir()
    assert train_module.is_finished(tmp_path) is False  # a checkpoint alone is an unfinished run
    (tmp_path / "adapter_model.safetensors").write_text("weights")
    assert train_module.is_finished(tmp_path) is False  # a checkpoint is left, so it resumes
    (tmp_path / "checkpoint-4").rmdir()
    assert train_module.is_finished(tmp_path) is True


def test_a_fresh_run_does_not_resume(train_module: types.ModuleType, tmp_path: Path) -> None:
    train_module.train(FakeModel(), FakeTokenizer(), [sample()], tmp_path, CONFIG)

    (trainer,) = FakeTrainer.instances
    assert trainer.resumed is False


def test_a_run_with_a_checkpoint_resumes_from_it(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "checkpoint-4").mkdir()

    train_module.train(FakeModel(), FakeTokenizer(), [sample()], tmp_path, CONFIG)

    (trainer,) = FakeTrainer.instances
    assert trainer.resumed is True


def test_checkpoints_are_removed_when_the_run_finishes_and_the_rest_is_kept(
    train_module: types.ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "checkpoint-4").mkdir()
    (tmp_path / "checkpoint-4" / "optimizer.pt").write_text("state")
    (tmp_path / "adapter_model.safetensors").write_text("weights")

    train_module.train(FakeModel(), FakeTokenizer(), [sample()], tmp_path, CONFIG)

    assert not list(tmp_path.glob("checkpoint-*"))
    assert (tmp_path / "adapter_model.safetensors").read_text() == "weights"
    assert (tmp_path / "training.log.json").exists()


LOGS = {"loss": 0.31654}


def test_progress_callback_shows_metrics_after_the_bar(train_module: types.ModuleType) -> None:
    callback = train_module.MetricsProgressCallback()
    callback.training_bar = FakeBar()
    callback.on_log(None, None, None, logs=LOGS)
    assert callback.training_bar.postfix == {"loss": "0.3165"}


def test_progress_callback_ignores_summary_and_missing_bar(train_module: types.ModuleType) -> None:
    callback = train_module.MetricsProgressCallback()
    callback.on_log(None, None, None, logs=LOGS)  # no bar yet
    callback.training_bar = FakeBar()
    callback.on_log(None, None, None, logs={"train_runtime": 1.0})
    assert callback.training_bar.postfix == {}
