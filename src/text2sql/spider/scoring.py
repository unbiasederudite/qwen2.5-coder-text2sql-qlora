"""Scoring of predictions with the official Spider evaluation code."""

import contextlib
import io
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nltk

from text2sql.spider.dataset import Example, Split, database_dir, load_examples, tables_file
from text2sql.spider.evaluation.evaluation import build_foreign_key_map_from_json, evaluate
from text2sql.spider.predictions import Prediction
from text2sql.spider.report import LEVELS, Report
from text2sql.spider.sql import extract_sql, one_line

_COMPONENTS = (  # the parts of a query that the official evaluation scores separately
    "select",
    "select(no AGG)",
    "where",
    "where(no OP)",
    "group(no Having)",
    "group",
    "order",
    "and/or",
    "IUEN",
    "keywords",
)
_EMPTY_PREDICTION = "SELECT"  # a blank line would end a session in the official file format

# The vendored functions have no annotations
_evaluate: Callable[..., dict[str, Any]] = evaluate
_build_kmaps: Callable[[str], dict[str, Any]] = build_foreign_key_map_from_json


def _align(predictions: list[Prediction], examples: list[Example]) -> list[str]:
    """Matches predictions to examples by id.

    Args:
        predictions (list[Prediction]): Predictions.
        examples (list[Example]): Examples of the split.

    Returns:
        list[str]: Predicted SQL in example order.

    Raises:
        ValueError: If the predictions do not match the examples.
    """
    by_id: dict[int, Prediction] = {}
    for prediction in predictions:
        if prediction["id"] in by_id:
            raise ValueError(f"duplicate id {prediction['id']}")
        by_id[prediction["id"]] = prediction
    expected = set(range(len(examples)))
    missing = sorted(expected - by_id.keys())
    unknown = sorted(by_id.keys() - expected)
    if missing or unknown:
        raise ValueError(
            f"ids do not match the {len(examples)} questions: "
            f"missing {missing[:5]}, unknown {unknown[:5]}"
        )
    for i, example in enumerate(examples):
        if by_id[i]["db_id"] != example["db_id"]:
            raise ValueError(f"id {i}: db_id {by_id[i]['db_id']} is not {example['db_id']}")
    return [by_id[i]["predicted"] for i in range(len(examples))]


def _run_official(
    gold: Path,
    predicted: Path,
    db_dir: Path,
    etype: str,
    kmaps: dict[str, Any] | None,
    keep_distinct: bool = False,
) -> dict[str, Any]:
    """Runs the official evaluation without its printing.

    Args:
        gold (Path): File of gold queries.
        predicted (Path): File of predicted queries.
        db_dir (Path): Directory of the databases.
        etype (str): Evaluation type, `all` or `exec`.
        kmaps (dict[str, Any] | None): Foreign key maps.
        keep_distinct (bool): Keep `DISTINCT`.

    Returns:
        dict[str, Any]: The official scores.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return _evaluate(
            str(gold), str(predicted), str(db_dir), etype, kmaps, False, keep_distinct, False
        )


def _write_official_files(
    directory: Path, examples: list[Example], predicted: list[str]
) -> tuple[Path, Path]:
    """Writes the gold and predicted queries in the format of the official evaluation.

    Args:
        directory (Path): Directory for the two files.
        examples (list[Example]): Examples of the split.
        predicted (list[str]): Predicted SQL in example order.

    Returns:
        tuple[Path, Path]: File of gold queries and file of predicted queries.
    """
    gold_file, predicted_file = directory / "gold.txt", directory / "predicted.txt"
    gold_file.write_text("".join(f"{one_line(e['query'])}\t{e['db_id']}\n" for e in examples))
    predicted_file.write_text(
        "".join(f"{one_line(sql) or _EMPTY_PREDICTION}\n" for sql in predicted)
    )
    return gold_file, predicted_file


def _build_report(full: dict[str, Any], ex: dict[str, Any], ts: dict[str, Any] | None) -> Report:
    """Picks the scores of a report out of the official results.

    Args:
        full (dict[str, Any]): Official results of the `all` evaluation.
        ex (dict[str, Any]): Official results of the execution on the original database.
        ts (dict[str, Any] | None): Official results on the test-suite databases, if any.

    Returns:
        Report: Scores by difficulty level.
    """
    return Report(
        count={level: full[level]["count"] for level in LEVELS},
        ex={level: float(ex[level]["exec"]) for level in LEVELS},
        ts=None if ts is None else {level: float(ts[level]["exec"]) for level in LEVELS},
        em={level: float(full[level]["exact"]) for level in LEVELS},
        cm={
            level: {
                component: {
                    metric: float(full[level]["partial"][component][metric])
                    for metric in ("acc", "rec", "f1")
                }
                for component in _COMPONENTS
            }
            for level in LEVELS
        },
    )


def score(
    predictions: list[Prediction],
    spider_dir: Path,
    split: Split,
    test_suite_dir: Path | None = None,
    keep_distinct: bool = False,
    extract: bool = True,
) -> Report:
    """Scores predictions against the gold SQL of one split.

    Args:
        predictions (list[Prediction]): Predictions, one per question.
        spider_dir (Path): Spider directory.
        split (Split): Split the predictions are for.
        test_suite_dir (Path | None): Test-suite databases, if any.
        keep_distinct (bool): Keep `DISTINCT` in EX and TS.
        extract (bool): Take the SQL out of each reply.

    Returns:
        Report: Scores by difficulty level.

    Raises:
        RuntimeError: If the tokenizer data cannot be downloaded.
        ValueError: If the predictions do not match the questions.
    """
    if not nltk.download("punkt_tab", quiet=True):  # no-op when already downloaded
        raise RuntimeError("could not download the nltk tokenizer data")
    examples = load_examples(spider_dir, split)
    predicted = _align(predictions, examples)
    if extract:
        predicted = [extract_sql(reply) for reply in predicted]
    kmaps = _build_kmaps(str(tables_file(spider_dir, split)))
    original = database_dir(spider_dir, split)
    with tempfile.TemporaryDirectory() as tmp:
        gold, pred = _write_official_files(Path(tmp), examples, predicted)
        if test_suite_dir is None:
            full = _run_official(gold, pred, original, "all", kmaps, keep_distinct)
            return _build_report(full, full, None)
        full = _run_official(gold, pred, test_suite_dir, "all", kmaps, keep_distinct)
        ex = _run_official(gold, pred, original, "exec", None, keep_distinct)
        return _build_report(full, ex, full)
