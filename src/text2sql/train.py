"""QLoRA fine-tuning on Spider."""

import shutil
from pathlib import Path
from typing import Any, TypedDict

import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_callback import ProgressCallback
from trl import SFTConfig, SFTTrainer

from text2sql.data import Sample
from text2sql.meta import write_json
from text2sql.prompt import build_completion, build_prompt
from text2sql.spider import Split

CHECKPOINTS = "checkpoint-*"  # the directories the trainer writes, one per epoch
ADAPTER_FILE = "adapter_model.safetensors"  # the weights, written when training is done


class TrainConfig(TypedDict):
    """One training config file."""

    splits: list[Split]  # Spider training splits to train on
    lora: dict[str, Any]  # arguments of `peft.LoraConfig`
    training: dict[
        str, Any
    ]  # arguments of `trl.SFTConfig`, with `max_length` also used for samples


class SampleCounts(TypedDict):
    """How many samples a training run used."""

    samples: int  # samples trained on
    dropped_too_long: int  # samples dropped for being longer than `max_length`


def load_config(path: Path) -> TrainConfig:
    """Reads a training config file and checks its values.

    Args:
        path (Path): YAML config file.

    Returns:
        TrainConfig: Splits, LoRA settings and training settings.

    Raises:
        ValueError: If a training setting is out of range.
    """
    config: TrainConfig = yaml.safe_load(path.read_text())
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


def has_checkpoint(adapter_dir: Path) -> bool:
    """Tells whether an adapter directory holds a checkpoint of an unfinished run.

    Args:
        adapter_dir (Path): Directory of the adapter.

    Returns:
        bool: True if a checkpoint is there.
    """
    return any(adapter_dir.glob(CHECKPOINTS))


def is_finished(adapter_dir: Path) -> bool:
    """Tells whether an adapter directory holds the adapter of a finished run.

    Args:
        adapter_dir (Path): Directory of the adapter.

    Returns:
        bool: True if the adapter is there and no checkpoint is left.
    """
    return (adapter_dir / ADAPTER_FILE).exists() and not has_checkpoint(adapter_dir)


def log_path(adapter_dir: Path) -> Path:
    """Names the training log file of an adapter.

    Args:
        adapter_dir (Path): Directory of the adapter.

    Returns:
        Path: `training.log.json` inside the directory.
    """
    return adapter_dir / "training.log.json"


def token_count(tokenizer: PreTrainedTokenizerBase, messages: list[Any]) -> int:
    """Counts the tokens of a conversation.

    Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer from `load_model`.
        messages (list[Any]): Conversation messages.

    Returns:
        int: Number of tokens.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_dataset(
    samples: list[Sample], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> tuple[Dataset, int]:
    """Builds the prompt-completion dataset.

    Args:
        samples (list[Sample]): Samples to train on.
        tokenizer (PreTrainedTokenizerBase): Tokenizer from `load_model`.
        max_length (int): Longest sample to keep, in tokens.

    Returns:
        tuple[Dataset, int]: Dataset, and the number of samples dropped for length.
    """
    rows = [{"prompt": build_prompt(s), "completion": build_completion(s)} for s in samples]
    kept = [
        row
        for row in rows
        if token_count(tokenizer, row["prompt"] + row["completion"]) <= max_length
    ]
    return Dataset.from_list(kept), len(rows) - len(kept)


class MetricsProgressCallback(ProgressCallback):  # type: ignore[misc]
    """Text progress bar that keeps one line and updates its loss on each log."""

    def on_log(
        self,
        args: object,
        state: object,
        control: object,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        """Shows the latest loss after the bar.

        Args:
            args (object): Training arguments, unused.
            state (object): Trainer state, unused.
            control (object): Trainer control, unused.
            logs (dict[str, float] | None): Metrics of the last logging step.
            **kwargs (object): Other callback arguments, unused.
        """
        if self.training_bar is None or not logs or "loss" not in logs:
            return
        self.training_bar.set_postfix(loss=f"{logs['loss']:.4f}")


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[Sample],
    adapter_dir: Path,
    config: TrainConfig,
) -> SampleCounts:
    """Fine-tunes the model with LoRA, resuming from a checkpoint if there is one.

    Args:
        model (PreTrainedModel): Model from `load_model`.
        tokenizer (PreTrainedTokenizerBase): Tokenizer from `load_model`.
        samples (list[Sample]): Samples to train on.
        adapter_dir (Path): Directory of the adapter.
        config (TrainConfig): Config from `load_config`.

    Returns:
        SampleCounts: Samples trained on and samples dropped for length.
    """
    dataset, dropped = build_dataset(samples, tokenizer, config["training"]["max_length"])
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(output_dir=str(adapter_dir), **config["training"]),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(**config["lora"]),
    )
    # TRL casts the adapter of a 4-bit model to bf16, which fp16 loss scaling cannot unscale
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    trainer.train(resume_from_checkpoint=has_checkpoint(adapter_dir))
    trainer.save_model(str(adapter_dir))
    write_json(log_path(adapter_dir), trainer.state.log_history)
    for checkpoint in adapter_dir.glob(CHECKPOINTS):
        shutil.rmtree(checkpoint)  # a finished run must not be resumed by the next one
    return SampleCounts(samples=len(dataset), dropped_too_long=dropped)
