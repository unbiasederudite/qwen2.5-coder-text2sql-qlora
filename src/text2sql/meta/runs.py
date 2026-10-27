"""The metadata files of generation, training and scoring runs."""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from text2sql.jsonio import write_json
from text2sql.meta.provenance import (
    ADAPTER_KEYS,
    adapter_info,
    attempt,
    environment,
    git_commit,
    model_info,
    repository,
    sha256,
    split_hash,
    timing,
)
from text2sql.spider.dataset import Split

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from text2sql.config import ModelConfig, TrainConfig
    from text2sql.generation import PredictionCounts
    from text2sql.training.train import SampleCounts


def metadata_path(path: Path) -> Path:
    """Names the metadata file of predictions, scores or an adapter directory.

    Args:
        path (Path): Predictions or scores file, or an existing adapter directory.

    Returns:
        Path: `<name>.meta.json`, or `training.meta.json` inside an adapter directory.
    """
    if path.is_dir():
        return path / "training.meta.json"
    return path.with_suffix(".meta.json")


def _warn(message: str) -> None:
    """Prints a warning to stderr.

    Args:
        message (str): Text of the warning.
    """
    print(f"warning: {message}", file=sys.stderr)


def _write_run_metadata(path: Path, info: dict[str, object]) -> None:
    """Writes the metadata of a run.

    Args:
        path (Path): Metadata file to write.
        info (dict[str, object]): Metadata of the run.
    """
    null_fields = [name for name, value in info.items() if value is None]
    if null_fields:
        _warn(f"fields that could not be read: {', '.join(null_fields)}")
    write_json(path, info)


def load_generation_metadata(predictions_path: Path, split: Split) -> dict[str, Any] | None:
    """Reads the metadata next to a predictions file and checks it against the scoring.

    Args:
        predictions_path (Path): Predictions file.
        split (Split): Split that is being scored.

    Returns:
        dict[str, Any] | None: Metadata, or None without a metadata file.

    Raises:
        ValueError: If the metadata describes another predictions file.
    """
    path = metadata_path(predictions_path)
    if not path.exists():
        return None
    info: dict[str, Any] = json.loads(path.read_text())
    if info.get("predictions_sha256") not in (None, sha256(predictions_path)):
        raise ValueError(f"{path} describes another predictions file")
    if info.get("split") not in (None, split):
        _warn(f"the metadata is for the {info['split']} split, scoring {split}")
    return info


def write_generation_metadata(
    predictions_path: Path,
    model: "PreTrainedModel",
    config: "ModelConfig",
    run: str,
    split: Split,
    batch_size: int,
    counts: "PredictionCounts",
    started_at: datetime,
    finished_at: datetime,
    adapter: str | None = None,
) -> Path:
    """Writes the metadata of a generation run next to its predictions file.

    Args:
        predictions_path (Path): Predictions file.
        model (PreTrainedModel): Model from `load_model`.
        config (ModelConfig): Config from `load_model_config`.
        run (str): Name of the run.
        split (Split): Split that was answered.
        batch_size (int): Questions generated at once.
        counts (PredictionCounts): Counts from `generate_predictions`.
        started_at (datetime): Start of the generation.
        finished_at (datetime): End of the generation.
        adapter (str | None): Adapter that was loaded on top of the model, if any.

    Returns:
        Path: Metadata file.
    """
    info = {
        "run": run,
        **model_info(model),
        **(adapter_info(adapter) if adapter else {}),
        "tokenizer": config["tokenizer"],
        "generation": {**config["generation"], "batch_size": batch_size},
        "split": split,
        "split_file_sha256": split_hash(split),
        "questions": counts["questions"],
        "already_done": counts["already_done"],
        "predictions_sha256": attempt(sha256, predictions_path),
        **timing(started_at, finished_at),
        **environment(("torch", "transformers", "bitsandbytes")),
    }
    path = metadata_path(predictions_path)
    _write_run_metadata(path, info)
    return path


def write_training_metadata(
    adapter_dir: Path,
    model: "PreTrainedModel",
    config: "TrainConfig",
    counts: "SampleCounts",
    resumed: bool,
    started_at: datetime,
    finished_at: datetime,
) -> Path:
    """Writes the metadata of a training run into the adapter directory.

    Args:
        adapter_dir (Path): Directory of the adapter.
        model (PreTrainedModel): Model from `load_model`.
        config (TrainConfig): Config from `load_train_config`.
        counts (SampleCounts): Sample counts from `train`.
        resumed (bool): Whether the run continued from a checkpoint.
        started_at (datetime): Start of the training.
        finished_at (datetime): End of the training.

    Returns:
        Path: Metadata file.
    """
    info = {
        **model_info(model),
        "data": config["data"],
        "lora": config["lora"],
        "training": config["training"],
        "split_files_sha256": {split: split_hash(split) for split in config["data"]["splits"]},
        "samples": counts["samples"],
        "dropped_too_long": counts["dropped_too_long"],
        "resumed": resumed,
        **timing(started_at, finished_at),
        **environment(("torch", "transformers", "bitsandbytes", "peft", "trl")),
    }
    path = metadata_path(adapter_dir)
    _write_run_metadata(path, info)
    return path


def write_scoring_metadata(
    report_path: Path,
    predictions_path: Path,
    split: Split,
    keep_distinct: bool,
    extraction: bool,
    generation_metadata: dict[str, Any] | None,
) -> Path:
    """Writes the metadata of a scoring run next to its scores file.

    Args:
        report_path (Path): Scores file.
        predictions_path (Path): Predictions file that was scored.
        split (Split): Split that was scored.
        keep_distinct (bool): Whether `DISTINCT` was kept in EX and TS.
        extraction (bool): Whether the SQL was taken out of the replies.
        generation_metadata (dict[str, Any] | None): Metadata of the predictions file, if any.

    Returns:
        Path: Metadata file.
    """
    given = generation_metadata or {}
    adapter = {key: given[key] for key in ADAPTER_KEYS if key in given}
    info = {
        "run": given.get("run"),
        "model": given.get("model"),
        "model_revision": given.get("model_revision"),
        **adapter,
        "split": split,
        "predictions_sha256": sha256(predictions_path),
        "keep_distinct": keep_distinct,
        "extraction": extraction,
        "repo": repository(),
        "commit": git_commit(),
        "sqlite_version": sqlite3.sqlite_version,
    }
    path = metadata_path(report_path)
    _write_run_metadata(path, info)
    return path
