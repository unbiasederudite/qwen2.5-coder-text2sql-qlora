"""Scoring of Spider predictions with the official evaluation code."""

import argparse
import contextlib
import io
import json
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import nltk

from text2sql.meta import read_generation_metadata, write_json, write_scoring_metadata
from text2sql.spider import SPIDER_DIR, Example, Split, load_examples
from text2sql.spider_evaluation.evaluation import build_foreign_key_map_from_json, evaluate

LEVELS = ("easy", "medium", "hard", "extra", "all")
COMPONENTS = (
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
TEST_SUITE_DIR = Path("data/test_suite_data")  # where `scripts/download_data.sh test-suite` puts it
EMPTY_PREDICTION = "SELECT"  # a blank line would end a session in the official file format
# The first markdown code block: an optional language tag (not SELECT or WITH), then the code
FENCE = re.compile(
    r"```[ \t]*(?:sql\b[ \t]*|(?!(?:select|with)\b)[\w+-]+[ \t]*\n)?(.*?)(?:```|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# The vendored functions have no annotations
_evaluate: Callable[..., dict[str, Any]] = evaluate
_build_kmaps: Callable[[str], dict[str, Any]] = build_foreign_key_map_from_json


class Prediction(TypedDict):
    """One line of a predictions file."""

    id: int  # position of the question in the split
    db_id: str  # database name
    predicted: str  # model output


class Report(TypedDict):
    """How a predictions file scored, by difficulty level and over all questions."""

    count: dict[str, int]  # questions
    ex: dict[str, float]  # execution accuracy on the original database
    ts: dict[str, float] | None  # test-suite accuracy, None without the test suite
    em: dict[str, float]  # exact set match
    cm: dict[str, dict[str, dict[str, float]]]  # level -> component -> acc, rec, f1


def load_predictions(path: Path) -> list[Prediction]:
    """Reads a predictions file (JSON Lines).

    Args:
        path (Path): Predictions file.

    Returns:
        list[Prediction]: Predictions in file order.

    Raises:
        ValueError: If a line is not a valid prediction.
    """
    predictions: list[Prediction] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            row = json.loads(line)
            prediction = Prediction(id=row["id"], db_id=row["db_id"], predicted=row["predicted"])
        except json.JSONDecodeError as e:
            raise ValueError(f"{where}: invalid JSON") from e
        except TypeError as e:
            raise ValueError(f"{where}: not a JSON object") from e
        except KeyError as e:
            raise ValueError(f"{where}: missing field {e}") from e
        is_int = isinstance(prediction["id"], int) and not isinstance(prediction["id"], bool)
        if not (
            is_int
            and isinstance(prediction["db_id"], str)
            and isinstance(prediction["predicted"], str)
        ):
            raise ValueError(f"{where}: id must be an integer, db_id and predicted strings")
        predictions.append(prediction)
    return predictions


def align(predictions: list[Prediction], examples: list[Example]) -> list[str]:
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


def one_line(sql: str) -> str:
    """Puts a query on one line.

    Args:
        sql (str): Query.

    Returns:
        str: Query without newlines and tabs.
    """
    parts = re.split(r"""('[^']*'|"[^"]*"|`[^`]*`)""", sql)  # odd parts are quoted
    return "".join(
        re.sub(r"\s", " ", part) if i % 2 else re.sub(r"\s+", " ", part)
        for i, part in enumerate(parts)
    ).strip()


def extract_sql(reply: str) -> str:
    """Extracts the SQL query from a model reply.

    Args:
        reply (str): Reply of the model.

    Returns:
        str: Query without the code fence and the text around it.
    """
    match = FENCE.search(reply)
    text = match.group(1) if match else reply
    quote = ""
    for i, char in enumerate(text):
        if quote:
            quote = "" if char == quote else quote
        elif char in "'\"`":
            quote = char
        elif char == ";":
            return text[:i].strip()
    return text.strip()


def run_official(
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
    predicted = align(predictions, examples)
    if extract:
        predicted = [extract_sql(reply) for reply in predicted]
    test = split == "test"
    kmaps = _build_kmaps(str(spider_dir / ("test_tables.json" if test else "tables.json")))
    original = spider_dir / ("test_database" if test else "database")
    with tempfile.TemporaryDirectory() as tmp:
        gold_file, predicted_file = Path(tmp) / "gold.txt", Path(tmp) / "predicted.txt"
        gold_file.write_text("".join(f"{one_line(e['query'])}\t{e['db_id']}\n" for e in examples))
        predicted_file.write_text(
            "".join(f"{one_line(sql) or EMPTY_PREDICTION}\n" for sql in predicted)
        )
        if test_suite_dir is None:
            full = run_official(gold_file, predicted_file, original, "all", kmaps, keep_distinct)
            ex, ts = full, None
        else:
            full = run_official(
                gold_file, predicted_file, test_suite_dir, "all", kmaps, keep_distinct
            )
            ex = run_official(gold_file, predicted_file, original, "exec", None, keep_distinct)
            ts = full
    return Report(
        count={level: full[level]["count"] for level in LEVELS},
        ex={level: float(ex[level]["exec"]) for level in LEVELS},
        ts=None if ts is None else {level: float(ts[level]["exec"]) for level in LEVELS},
        em={level: float(full[level]["exact"]) for level in LEVELS},
        cm={
            level: {
                component: {
                    "acc": float(full[level]["partial"][component]["acc"]),
                    "rec": float(full[level]["partial"][component]["rec"]),
                    "f1": float(full[level]["partial"][component]["f1"]),
                }
                for component in COMPONENTS
            }
            for level in LEVELS
        },
    )


def report_path(predictions_path: Path) -> Path:
    """Names the scores file of a predictions file.

    Args:
        predictions_path (Path): Predictions file.

    Returns:
        Path: `<name>.report.json` in the `scores/` directory.
    """
    return Path("scores") / f"{predictions_path.stem}.report.json"


def write_report(report: Report, path: Path) -> None:
    """Writes the scores of a report.

    Args:
        report (Report): Report from `score`.
        path (Path): File for the scores.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report)


def format_report(report: Report) -> str:
    """Formats the report as a table of accuracies by difficulty level.

    Args:
        report (Report): Report to format.

    Returns:
        str: Table, one row per metric.
    """
    rows = {"EX": report["ex"], "TS": report["ts"], "EM": report["em"]}
    lines = [f"{'':6}" + "".join(f"{level:>9}" for level in LEVELS)]
    lines.append(f"{'count':6}" + "".join(f"{report['count'][level]:>9}" for level in LEVELS))
    for name, values in rows.items():
        cells = ["      -" if values is None else f"{values[level]:.3f}" for level in LEVELS]
        lines.append(f"{name:6}" + "".join(f"{cell:>9}" for cell in cells))
    return "\n".join(lines)


def main() -> None:
    """Scores a predictions file, prints the table and writes the scores and metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="predictions file (JSON Lines)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--spider-dir", type=Path, default=SPIDER_DIR)
    parser.add_argument(
        "--test-suite-dir",
        type=Path,
        default=TEST_SUITE_DIR,
        help="test-suite databases, used for the dev split only",
    )
    parser.add_argument(
        "--keep-distinct",
        action="store_true",
        help="keep DISTINCT in EX and TS, the official script strips it (EM and CM always ignore it)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="score the replies as written, without taking the SQL out of code blocks",
    )
    args = parser.parse_args()

    suite = args.test_suite_dir if args.split == "dev" else None
    if suite is not None and not suite.exists():
        parser.error(f"{suite} not found, download it: bash scripts/download_data.sh test-suite")
    try:
        generation_metadata = read_generation_metadata(args.predictions, args.split)
    except ValueError as error:
        parser.error(str(error))
    report = score(
        load_predictions(args.predictions),
        args.spider_dir,
        args.split,
        suite,
        args.keep_distinct,
        not args.raw,
    )
    print(format_report(report))
    if args.keep_distinct:
        print("DISTINCT kept in EX and TS")
    if args.raw:
        print("replies scored as written, without extraction")
    report_file = report_path(args.predictions)
    write_report(report, report_file)
    written = write_scoring_metadata(
        report_file,
        args.predictions,
        args.split,
        args.keep_distinct,
        not args.raw,
        generation_metadata,
    )
    print(f"scores in {report_file}, metadata in {written}")


if __name__ == "__main__":
    main()
