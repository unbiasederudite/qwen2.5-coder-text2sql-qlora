"""Tests for text2sql.generate, with a stand-in for the model (no GPU or install needed)."""

import json
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from text2sql.data import Sample
from text2sql.prompt import Message

QUESTIONS = [
    "a long question about many singers",  # 0, the longest prompt
    "short",  # 1
    "medium question",  # 2
    "tiny",  # 3, the shortest prompt
    "a somewhat longer question",  # 4
]
SHORTEST_FIRST = [3, 1, 2, 4, 0]
SAMPLES = [
    Sample(db_id=f"db{i}", question=q, query="SELECT 1", schema=["CREATE TABLE t (x INT);"])
    for i, q in enumerate(QUESTIONS)
]


class Recorder:
    """Stands in for `generate_batch`: records the questions of each call and answers them."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.fail_on_call: int | None = None
        self.watched: Path | None = None  # a file whose lines are counted at every call
        self.lines_seen: list[int] = []

    def __call__(
        self, model: object, tokenizer: object, conversations: list[list[Message]]
    ) -> list[str]:
        if self.watched is not None:
            self.lines_seen.append(len(self.watched.read_text().splitlines()))
        if self.fail_on_call == len(self.batches) + 1:
            raise RuntimeError("out of memory")
        questions = [conversation[-1]["content"] for conversation in conversations]
        self.batches.append(questions)
        return [f"SQL for {question}" for question in questions]


@pytest.fixture
def recorder(stub_module: Callable[..., types.ModuleType]) -> Recorder:
    recorder = Recorder()
    stub_module("text2sql.model", generate_batch=recorder)
    stub_module("transformers", PreTrainedModel=object, PreTrainedTokenizerBase=object)
    return recorder


@pytest.fixture
def generate_module(
    recorder: Recorder, fresh_import: Callable[[str], types.ModuleType]
) -> types.ModuleType:
    """`text2sql.generate` imported against the stand-ins."""
    return fresh_import("generate")


def run(module: types.ModuleType, path: Path, batch_size: int = 2) -> dict[str, int]:
    counts: dict[str, int] = module.generate_predictions(
        object(), object(), SAMPLES, path, batch_size
    )
    return counts


def read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def line(i: int, predicted: str) -> str:
    return json.dumps({"id": i, "db_id": f"db{i}", "predicted": predicted}) + "\n"


def test_each_prediction_is_written_under_the_id_and_database_of_its_question(
    generate_module: types.ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "predictions.jsonl"

    run(generate_module, path)

    by_id = {row["id"]: row for row in read(path)}
    assert by_id == {
        i: {"id": i, "db_id": f"db{i}", "predicted": f"SQL for {q}"}
        for i, q in enumerate(QUESTIONS)
    }
    assert len(read(path)) == len(QUESTIONS)  # one line per question


def test_questions_are_answered_in_batches_shortest_prompt_first(
    generate_module: types.ModuleType, recorder: Recorder, tmp_path: Path
) -> None:
    run(generate_module, tmp_path / "p.jsonl", batch_size=2)

    ordered = [QUESTIONS[i] for i in SHORTEST_FIRST]
    assert recorder.batches == [ordered[0:2], ordered[2:4], ordered[4:5]]  # the last is smaller


def test_the_counts_say_how_many_were_generated_and_how_many_were_there(
    generate_module: types.ModuleType, tmp_path: Path
) -> None:
    assert run(generate_module, tmp_path / "p.jsonl") == {"generated": 5, "already_done": 0}


def test_questions_already_in_the_file_are_skipped_and_the_file_is_appended_to(
    generate_module: types.ModuleType, recorder: Recorder, tmp_path: Path
) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text(line(1, "old one") + line(3, "old three"))

    counts = run(generate_module, path)

    assert counts == {"generated": 3, "already_done": 2}
    asked = {question for batch in recorder.batches for question in batch}
    assert asked == {QUESTIONS[0], QUESTIONS[2], QUESTIONS[4]}
    rows = {row["id"]: row["predicted"] for row in read(path)}
    assert rows[1] == "old one" and rows[3] == "old three"  # kept as they were
    assert sorted(row["id"] for row in read(path)) == [0, 1, 2, 3, 4]  # each question once


def test_nothing_is_generated_when_every_question_is_in_the_file(
    generate_module: types.ModuleType, recorder: Recorder, tmp_path: Path
) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text("".join(line(i, "old") for i in range(5)))

    counts = run(generate_module, path)

    assert counts == {"generated": 0, "already_done": 5}
    assert recorder.batches == []
    assert path.read_text() == "".join(line(i, "old") for i in range(5))


def test_each_batch_is_on_disk_before_the_next_one_starts(
    generate_module: types.ModuleType, recorder: Recorder, tmp_path: Path
) -> None:
    path = tmp_path / "p.jsonl"
    recorder.watched = path

    run(generate_module, path, batch_size=2)

    assert recorder.lines_seen == [0, 2, 4]  # what a killed process would have left


def test_a_failed_batch_keeps_the_finished_ones_and_a_rerun_resumes_from_them(
    generate_module: types.ModuleType, recorder: Recorder, tmp_path: Path
) -> None:
    path = tmp_path / "p.jsonl"
    recorder.fail_on_call = 2  # the first batch is done, the second fails

    with pytest.raises(RuntimeError, match="out of memory"):
        run(generate_module, path, batch_size=2)

    assert sorted(row["id"] for row in read(path)) == sorted(SHORTEST_FIRST[:2])
    recorder.fail_on_call = None
    recorder.batches.clear()

    counts = run(generate_module, path, batch_size=2)

    assert counts == {"generated": 3, "already_done": 2}
    assert sorted(row["id"] for row in read(path)) == [0, 1, 2, 3, 4]


def test_the_directory_of_the_predictions_file_is_created(
    generate_module: types.ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "predictions" / "p.jsonl"

    run(generate_module, path)

    assert path.exists()
