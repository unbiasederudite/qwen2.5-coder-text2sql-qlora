"""The checkpoints of a training run, and what they say about it: unfinished or finished."""

import shutil
from pathlib import Path

ADAPTER_FILE = "adapter_model.safetensors"  # the weights of the adapter, saved when the run ends
_CHECKPOINTS = "checkpoint-*"  # the directories the trainer writes while it runs


def has_checkpoint(adapter_dir: Path) -> bool:
    """Tells whether an adapter directory holds a checkpoint of an unfinished run.

    Args:
        adapter_dir (Path): Directory of the adapter.

    Returns:
        bool: True if a checkpoint is there.
    """
    return any(adapter_dir.glob(_CHECKPOINTS))


def is_finished(adapter_dir: Path) -> bool:
    """Tells whether an adapter directory holds the adapter of a finished run.

    Args:
        adapter_dir (Path): Directory of the adapter.

    Returns:
        bool: True if the adapter is there and no checkpoint is left.
    """
    return (adapter_dir / ADAPTER_FILE).exists() and not has_checkpoint(adapter_dir)


def remove_checkpoints(adapter_dir: Path) -> None:
    """Removes the checkpoints, so that the next run does not resume a finished one.

    Args:
        adapter_dir (Path): Directory of the adapter.
    """
    for checkpoint in adapter_dir.glob(_CHECKPOINTS):
        shutil.rmtree(checkpoint)
