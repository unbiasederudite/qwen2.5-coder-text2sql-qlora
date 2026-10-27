"""Tests for text2sql.training.checkpoints."""

from pathlib import Path

from text2sql.training import checkpoints


def test_has_checkpoint_is_true_only_with_a_checkpoint_in_the_directory(tmp_path: Path) -> None:
    assert checkpoints.has_checkpoint(tmp_path) is False
    (tmp_path / "adapter_config.json").write_text("{}")
    assert checkpoints.has_checkpoint(tmp_path) is False
    (tmp_path / "checkpoint-4").mkdir()
    assert checkpoints.has_checkpoint(tmp_path) is True


def test_a_run_is_finished_only_with_the_adapter_and_no_checkpoint(tmp_path: Path) -> None:
    assert checkpoints.is_finished(tmp_path) is False  # nothing there
    (tmp_path / "checkpoint-4").mkdir()
    assert checkpoints.is_finished(tmp_path) is False  # a checkpoint alone is an unfinished run
    (tmp_path / "adapter_model.safetensors").write_text("weights")
    assert checkpoints.is_finished(tmp_path) is False  # a checkpoint is left, so it resumes
    (tmp_path / "checkpoint-4").rmdir()
    assert checkpoints.is_finished(tmp_path) is True


def test_remove_checkpoints_keeps_the_rest_of_the_directory(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-4").mkdir()
    (tmp_path / "checkpoint-4" / "optimizer.pt").write_text("state")
    (tmp_path / "checkpoint-8").mkdir()
    (tmp_path / checkpoints.ADAPTER_FILE).write_text("weights")

    checkpoints.remove_checkpoints(tmp_path)

    assert [p.name for p in tmp_path.iterdir()] == [checkpoints.ADAPTER_FILE]
