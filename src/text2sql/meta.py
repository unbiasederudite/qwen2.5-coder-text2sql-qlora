"""Metadata of prediction runs and of their scores."""

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

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def attempt(function: Callable[..., object], *args: object) -> object | None:
    """Returns `function(*args)`, or None if it raises."""
    try:
        return function(*args)
    except Exception:  # noqa: BLE001
        return None


def sha256(path: Path) -> str:
    """Returns the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_revision(model_name: str) -> str:
    """Returns the commit of the cached Hugging Face model."""
    from huggingface_hub import try_to_load_from_cache

    # a cached file lives in snapshots/<commit>/, so its folder name is the commit
    return Path(str(try_to_load_from_cache(model_name, "config.json"))).parent.name


def git(*args: str) -> str | None:
    """Runs a git command and returns its output.

    Args:
        *args (str): Arguments of git.

    Returns:
        str | None: Output without the trailing newline, or None if the command fails.
    """
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit() -> str | None:
    """Returns the commit of the checked-out code, or None outside a git repository."""
    return git("rev-parse", "HEAD")


def repository() -> str | None:
    """Returns the GitHub repository of the code as `owner/name`, from the `origin` remote."""
    url = git("remote", "get-url", "origin")
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url or "")
    return match.group(1) if match else None


def metadata_path(path: Path) -> Path:
    """Returns the metadata file of a predictions or scores file, next to it."""
    return path.with_suffix(".meta.json")


def to_json(value: object) -> object:
    """Serializes the config objects that `json` cannot."""
    return value.to_dict() if hasattr(value, "to_dict") else str(value)


def write_json(path: Path, data: dict[str, object]) -> None:
    """Writes a metadata file."""
    path.write_text(json.dumps(data, indent=2, default=to_json) + "\n")


def read_generation_metadata(predictions_path: Path, split: str) -> dict[str, Any] | None:
    """Reads the metadata file next to a predictions file and checks it against the scoring.

    Args:
        predictions_path (Path): Predictions file.
        split (str): Split that is being scored. A different split in the metadata only warns.

    Returns:
        dict[str, Any] | None: Metadata, or None if there is no `.meta.json` file next to it.

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
        print(
            f"warning: the metadata is for the {info['split']} split, scoring {split}",
            file=sys.stderr,
        )
    return info


def package_versions() -> dict[str, object]:
    """Returns the installed versions of the packages that affect generation."""
    return {
        name: attempt(metadata.version, name) for name in ("torch", "transformers", "bitsandbytes")
    }


def gpu_name() -> str:
    """Returns the name of the first GPU."""
    import torch

    return str(torch.cuda.get_device_name(0))


def write_generation_metadata(
    predictions_path: Path,
    model: "PreTrainedModel",
    run: str,
    split: str,
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
        split (str): Split that was answered.
        batch_size (int): Questions generated at once, recorded with the generation settings.
        questions (int): Number of questions in the split.
        already_done (int): Predictions already in the file when the run started.
        started_at (datetime): Start of the generation.
        finished_at (datetime): End of the generation.

    Returns:
        Path: Metadata file, named like the predictions file with the suffix `.meta.json`.
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
        "split_file_sha256": attempt(sha256, Path(f"data/spider_data/{split}.json")),
        "questions": questions,
        "already_done": already_done,
        "predictions_sha256": attempt(sha256, predictions_path),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": round((finished_at - started_at).total_seconds()),
        "repo": repository(),
        "commit": git_commit(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "gpu": attempt(gpu_name),
    }
    null_fields = [name for name, value in info.items() if value is None]
    if null_fields:
        print(f"fields that could not be read: {', '.join(null_fields)}")
    path = metadata_path(predictions_path)
    write_json(path, info)
    return path


def write_scoring_metadata(
    report_path: Path,
    predictions_path: Path,
    split: str,
    keep_distinct: bool,
    extraction: bool,
    generation_metadata: dict[str, Any] | None,
) -> Path:
    """Writes the metadata of a scores file next to it.

    Args:
        report_path (Path): Scores file.
        predictions_path (Path): Predictions file that was scored.
        split (str): Split that was scored.
        keep_distinct (bool): `DISTINCT` was kept in EX and TS.
        extraction (bool): The SQL was taken out of the replies, not scored as written.
        generation_metadata (dict[str, Any] | None): Metadata from `read_generation_metadata`,
            the source of the model, its revision and the run.

    Returns:
        Path: Metadata file, named like the scores file with the suffix `.meta.json`.
    """
    given = generation_metadata or {}
    path = metadata_path(report_path)
    write_json(
        path,
        {
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
        },
    )
    return path
