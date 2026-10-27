"""Where a run came from: the code, the packages, the machine, the time, and the model, adapter and data it used."""

import hashlib
import platform
import re
import subprocess
from collections.abc import Callable
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from text2sql.spider.dataset import SPIDER_DIR, Split, split_file
from text2sql.training.checkpoints import ADAPTER_FILE

if TYPE_CHECKING:
    from transformers import PreTrainedModel

ADAPTER_KEYS = ("adapter", "adapter_sha256", "adapter_revision")  # the keys of `adapter_info`


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


def _git(*args: str) -> str | None:
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
    return _git("rev-parse", "HEAD")


def repository() -> str | None:
    """Finds the GitHub repository of the code.

    Returns:
        str | None: `owner/name`, or None without a GitHub remote.
    """
    url = _git("remote", "get-url", "origin")
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url or "")
    return match.group(1) if match else None


def _cached_revision(model_name: str, filename: str = "config.json") -> str:
    """Finds the commit of a cached Hugging Face model.

    Args:
        model_name (str): Hugging Face model name.
        filename (str): A file of the model that is in the cache.

    Returns:
        str: Commit of the cached model.

    Raises:
        FileNotFoundError: If the file is not in the cache.
    """
    from huggingface_hub import try_to_load_from_cache  # not installed with the eval group

    path = try_to_load_from_cache(model_name, filename)
    if not isinstance(path, str):
        raise FileNotFoundError(f"{model_name}/{filename} is not in the cache")
    # A cached file lives in snapshots/<commit>/, so its directory name is the commit
    return Path(path).parent.name


def _gpu_name() -> str:
    """Finds the name of the first GPU.

    Returns:
        str: Name of the first GPU.
    """
    import torch  # installed only on Colab

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
        "packages": {name: attempt(metadata.version, name) for name in packages},
        "gpu": attempt(_gpu_name),
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


def model_info(model: "PreTrainedModel") -> dict[str, object]:
    """Identifies a loaded model.

    Args:
        model (PreTrainedModel): Model from `load_model`.

    Returns:
        dict[str, object]: Name, cached commit and quantization config.
    """
    name = model.config.name_or_path
    return {
        "model": name,
        "model_revision": attempt(_cached_revision, name),
        "quantization": getattr(model.config, "quantization_config", None),
    }


def adapter_info(adapter: str) -> dict[str, object]:
    """Identifies a LoRA adapter.

    Args:
        adapter (str): Adapter directory, or Hugging Face repository of an adapter.

    Returns:
        dict[str, object]: The adapter, and the hash of its weights if it is a directory or
            its commit if it is a repository.
    """
    path = Path(adapter)
    if path.is_dir():
        return {"adapter": adapter, "adapter_sha256": attempt(sha256, path / ADAPTER_FILE)}
    return {
        "adapter": adapter,
        "adapter_revision": attempt(_cached_revision, adapter, "adapter_config.json"),
    }


def split_hash(split: Split) -> object | None:
    """Hashes the questions file of a split.

    Args:
        split (Split): Split to hash.

    Returns:
        object | None: SHA-256 of the file, or None if it cannot be read.
    """
    return attempt(sha256, split_file(SPIDER_DIR, split))
