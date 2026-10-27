"""Scores Spider predictions with the official evaluation code: `python -m text2sql.score`."""

import argparse
from pathlib import Path

from text2sql.jsonio import write_json
from text2sql.meta.runs import load_generation_metadata, write_scoring_metadata
from text2sql.spider.dataset import SPIDER_DIR, TEST_SUITE_DIR
from text2sql.spider.predictions import load_predictions
from text2sql.spider.report import format_report
from text2sql.spider.scoring import score


def _build_parser() -> argparse.ArgumentParser:
    """Defines the command line.

    Returns:
        argparse.ArgumentParser: Parser of the arguments.
    """
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
    return parser


def main() -> None:
    """Scores a predictions file, prints the table and writes the scores and metadata."""
    parser = _build_parser()
    args = parser.parse_args()
    test_suite_dir = args.test_suite_dir if args.split == "dev" else None
    if test_suite_dir is not None and not test_suite_dir.exists():
        parser.error(
            f"{test_suite_dir} not found, download it: bash scripts/download_data.sh test-suite"
        )
    try:
        generation_metadata = load_generation_metadata(args.predictions, args.split)
    except ValueError as error:
        parser.error(str(error))

    report = score(
        load_predictions(args.predictions),
        args.spider_dir,
        args.split,
        test_suite_dir,
        args.keep_distinct,
        not args.raw,
    )
    print(format_report(report))
    if args.keep_distinct:
        print("DISTINCT kept in EX and TS")
    if args.raw:
        print("replies scored as written, without extraction")

    report_file = Path("scores") / f"{args.predictions.stem}.report.json"
    report_file.parent.mkdir(exist_ok=True)
    write_json(report_file, report)
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
