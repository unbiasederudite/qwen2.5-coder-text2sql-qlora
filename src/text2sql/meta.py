"""Metadata of prediction runs, training runs and scores."""

import hashlib
import json
import platform
import re
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from text2sql.spider import SPIDER_DIR, Split, split_file

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from text2sql.train import SampleCounts, TrainConfig


def attempt(function: Callable[..., object], *args: object) -> object | None:
    """Calls a function and returns None if it raises.

    Args:
        function (Callable[..., object]): Function to call.
        *args (object): Arguments of the call.

    Returns:
        object | None: Result of `function(*args)`, or None if it raises.
    """
    try:
        return function(*args)
    except Exception:  # noqa: BLE001
        return None


def sha256(path: Path) -> str:
    """Hashes a file.

    Args:
        path (Path): File to hash.

    Returns:
        str: SHA-256 of the file content.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_revision(model_name: str) -> str:
    """Finds the commit of a cached Hugging Face model.

    Args:
        model_name (str): Hugging Face model name.

    Returns:
        str: Commit of the cached model.
    """
    from huggingface_hub import try_to_load_from_cache

    # A cached file lives in snapshots/<commit>/, so its directory name is the commit
    return Path(str(try_to_load_from_cache(model_name, "config.json"))).parent.name


def git(*args: str) -> str | None:
    """Runs a git command and returns its output.

    Args:
        *args (str): Arguments of git.

    Returns:
        str | None: Output, or None if the command fails.
    """
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit() -> str | None:
    """Finds the commit of the checked-out code.

    Returns:
        str | None: Commit hash, or None outside a git repository.
    """
    return git("rev-parse", "HEAD")


def repository() -> str | None:
    """Finds the GitHub repository of the code.

    Returns:
        str | None: `owner/name`, or None without a GitHub remote.
    """
    url = git("remote", "get-url", "origin")
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url or "")
    return match.group(1) if match else None


def metadata_path(path: Path) -> Path:
    """Names the metadata file of predictions, scores or an adapter directory.

    Args:
        path (Path): Predictions or scores file, or an existing adapter directory.

    Returns:
        Path: `<name>.meta.json`, or `training.meta.json` inside a directory.
    """
    if path.is_dir():
        return path / "training.meta.json"
    return path.with_suffix(".meta.json")


def to_json(value: object) -> object:
    """Serializes an object that `json` cannot.

    Args:
        value (object): Object to serialize.

    Returns:
        object: Its `to_dict()`, or its string.
    """
    return value.to_dict() if hasattr(value, "to_dict") else str(value)


def write_json(path: Path, data: object) -> None:
    """Writes data as indented JSON.

    Args:
        path (Path): File to write.
        data (object): Data to write.
    """
    path.write_text(json.dumps(data, indent=2, default=to_json) + "\n")


def warn(message: str) -> None:
    """Prints a warning to stderr.

    Args:
        message (str): Text of the warning.
    """
    print(f"warning: {message}", file=sys.stderr)


def read_generation_metadata(predictions_path: Path, split: Split) -> dict[str, Any] | None:
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
        warn(f"the metadata is for the {info['split']} split, scoring {split}")
    return info


def package_versions(names: tuple[str, ...]) -> dict[str, object]:
    """Looks up the installed versions of packages.

    Args:
        names (tuple[str, ...]): Package names.

    Returns:
        dict[str, object]: Version of each package, or None.
    """
    return {name: attempt(metadata.version, name) for name in names}


def gpu_name() -> str:
    """Finds the name of the first GPU.

    Returns:
        str: Name of the first GPU.
    """
    import torch

    return str(torch.cuda.get_device_name(0))


def environment(packages: tuple[str, ...]) -> dict[str, object]:
    """Describes the code and the machine of a run.

    Args:
        packages (tuple[str, ...]): Packages whose versions affect the run.

    Returns:
        dict[str, object]: Repository, commit, Python version, package versions and GPU.
    """
    return {
        "repo": repository(),
        "commit": git_commit(),
        "python": platform.python_version(),
        "packages": package_versions(packages),
        "gpu": attempt(gpu_name),
    }


def timing(started_at: datetime, finished_at: datetime) -> dict[str, object]:
    """Describes when a run started and finished, and how long it took.

    Args:
        started_at (datetime): Start of the run.
        finished_at (datetime): End of the run.

    Returns:
        dict[str, object]: Start, end and duration in seconds.
    """
    return {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": round((finished_at - started_at).total_seconds()),
    }


def write_run_metadata(path: Path, info: dict[str, object]) -> None:
    """Writes the metadata of a run.

    Args:
        path (Path): Metadata file to write.
        info (dict[str, object]): Metadata of the run.
    """
    null_fields = [name for name, value in info.items() if value is None]
    if null_fields:
        warn(f"fields that could not be read: {', '.join(null_fields)}")
    write_json(path, info)


def write_generation_metadata(
    predictions_path: Path,
    model: "PreTrainedModel",
    run: str,
    split: Split,
    batch_size: int,
    questions: int,
    already_done: int,
    started_at: datetime,
    finished_at: datetime,
) -> Path:
    """Writes the metadata of a run next to its predictions file.

    Args:
        predictions_path (Path): Predictions file.
        model (PreTrainedModel): Model from `load_model`.
        run (str): Name of the run.
        split (Split): Split that was answered.
        batch_size (int): Questions generated at once.
        questions (int): Number of questions in the split.
        already_done (int): Predictions already in the file when the run started.
        started_at (datetime): Start of the generation.
        finished_at (datetime): End of the generation.

    Returns:
        Path: Metadata file.
    """
    from text2sql.model import GENERATION

    model_name = model.config.name_or_path
    info = {
        "run": run,
        "model": model_name,
        "model_revision": attempt(cached_revision, model_name),
        "quantization": getattr(model.config, "quantization_config", None),
        "generation": {**GENERATION, "batch_size": batch_size},
        "split": split,
        "split_file_sha256": attempt(sha256, split_file(SPIDER_DIR, split)),
        "questions": questions,
        "already_done": already_done,
        "predictions_sha256": attempt(sha256, predictions_path),
        **timing(started_at, finished_at),
        **environment(("torch", "transformers", "bitsandbytes")),
    }
    path = metadata_path(predictions_path)
    write_run_metadata(path, info)
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
        config (TrainConfig): Config from `load_config`.
        counts (SampleCounts): Sample counts from `train`.
        resumed (bool): Whether the run continued from a checkpoint.
        started_at (datetime): Start of the training.
        finished_at (datetime): End of the training.

    Returns:
        Path: Metadata file.
    """
    model_name = model.config.name_or_path
    info = {
        "model": model_name,
        "model_revision": attempt(cached_revision, model_name),
        "quantization": getattr(model.config, "quantization_config", None),
        "lora": config["lora"],
        "training": config["training"],
        "splits": config["splits"],
        "split_files_sha256": {
            split: attempt(sha256, split_file(SPIDER_DIR, split)) for split in config["splits"]
        },
        "samples": counts["samples"],
        "dropped_too_long": counts["dropped_too_long"],
        "resumed": resumed,
        **timing(started_at, finished_at),
        **environment(("torch", "transformers", "bitsandbytes", "peft", "trl")),
    }
    path = metadata_path(adapter_dir)
    write_run_metadata(path, info)
    return path


def write_scoring_metadata(
    report_path: Path,
    predictions_path: Path,
    split: Split,
    keep_distinct: bool,
    extraction: bool,
    generation_metadata: dict[str, Any] | None,
) -> Path:
    """Writes the metadata of a scores file next to it.

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
    info = {
        "run": given.get("run"),
        "model": given.get("model"),
        "model_revision": given.get("model_revision"),
        "split": split,
        "predictions_sha256": sha256(predictions_path),
        "keep_distinct": keep_distinct,
        "extraction": extraction,
        "repo": repository(),
        "commit": git_commit(),
        "sqlite_version": sqlite3.sqlite_version,
    }
    path = metadata_path(report_path)
    write_run_metadata(path, info)
    return path
