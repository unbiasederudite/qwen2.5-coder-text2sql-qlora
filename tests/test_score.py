import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

nltk = pytest.importorskip("nltk")
if not nltk.download("punkt_tab", quiet=True):  # no-op when already downloaded
    pytest.skip("could not download the nltk tokenizer data", allow_module_level=True)

from text2sql import score as score_module
from text2sql.score import (
    Prediction,
    align,
    extract_sql,
    format_report,
    load_predictions,
    main,
    one_line,
    score,
    write_report,
)
from text2sql.spider import Example

TABLES = [
    {
        "db_id": "concert",
        "table_names_original": ["singer"],
        "column_names_original": [[-1, "*"], [0, "singer_id"], [0, "name"], [0, "age"]],
        "column_types": ["text", "number", "text", "number"],
        "primary_keys": [1],
        "foreign_keys": [],
    }
]

ALL_AGE = "SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)"
# (gold SQL, predicted SQL, difficulty), in dev.json order
QUESTIONS = [
    # same query: right everywhere
    ("SELECT count(*) FROM singer", "SELECT count(*) FROM singer", "easy"),
    # 3 singers in the original database but 4 in the other version: right only by luck
    ("SELECT count(*) FROM singer", "SELECT 3", "easy"),
    (
        "SELECT name FROM singer WHERE age > 35 ORDER BY age",
        "SELECT name FROM singer WHERE age > 35 ORDER BY age",
        "medium",
    ),
    # the average age is 41.3 here (only Joe is older) but 36 in the other version
    (ALL_AGE, "SELECT name FROM singer WHERE age > 40", "hard"),
    (
        "SELECT name FROM singer WHERE age > 35 AND name LIKE 'T%' ORDER BY age LIMIT 1",
        "SELEC name",
        "extra",
    ),
    # the gold returns no rows, and an empty prediction must still be wrong
    (ALL_AGE + " AND name LIKE 'T%' ORDER BY age", "", "extra"),
]
LEVELS = ["easy", "medium", "hard", "extra", "all"]


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs every test in its own folder, where the default `scores/` folder is created."""
    monkeypatch.chdir(tmp_path)


def read_scores(tmp_path: Path, name: str = "run") -> tuple[dict[str, Any], dict[str, Any]]:
    """Reads the scores and the metadata of a report written to the default folder."""
    scores = json.loads((tmp_path / "scores" / f"{name}.report.json").read_text())
    metadata = json.loads((tmp_path / "scores" / f"{name}.report.meta.json").read_text())
    return scores, metadata


def make_database(path: Path, extra_singer: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(1, "Joe", 52), (2, "Tim", 32), (3, "Ann", 40)] + (
        [(4, "Zed", 20)] if extra_singer else []
    )
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE singer (singer_id int PRIMARY KEY, name text, age int)")
    con.executemany("INSERT INTO singer VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()


@pytest.fixture
def spider_dir(tmp_path: Path) -> Path:
    root = tmp_path / "spider"
    root.mkdir()
    dev = [{"db_id": "concert", "question": "?", "query": gold} for gold, _, _ in QUESTIONS]
    (root / "dev.json").write_text(json.dumps(dev))
    (root / "tables.json").write_text(json.dumps(TABLES))
    make_database(root / "database" / "concert" / "concert.sqlite", extra_singer=False)
    return root


@pytest.fixture
def suite_dir(tmp_path: Path) -> Path:
    root = tmp_path / "suite"
    make_database(root / "concert" / "concert.sqlite", extra_singer=False)  # the original
    make_database(root / "concert" / "concert_v2.sqlite", extra_singer=True)
    return root


@pytest.fixture
def predictions() -> list[Prediction]:
    return [
        Prediction(id=i, db_id="concert", predicted=predicted)
        for i, (_, predicted, _) in enumerate(QUESTIONS)
    ]


def test_load_predictions_reads_json_lines_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text(
        '{"id": 0, "db_id": "a", "predicted": "SELECT 1"}\n\n{"id": 1, "db_id": "b", "predicted": ""}\n'
    )

    assert load_predictions(path) == [
        {"id": 0, "db_id": "a", "predicted": "SELECT 1"},
        {"id": 1, "db_id": "b", "predicted": ""},
    ]


def test_load_predictions_names_the_line_with_a_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text('{"id": 0, "db_id": "a", "predicted": "x"}\n{"id": 1, "db_id": "a"}\n')

    with pytest.raises(ValueError, match=r"p\.jsonl:2: missing field 'predicted'"):
        load_predictions(path)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "invalid JSON"),
        ("[1, 2]", "not a JSON object"),
        ('{"id": "0", "db_id": "a", "predicted": "x"}', "id must be an integer"),
        ('{"id": true, "db_id": "a", "predicted": "x"}', "id must be an integer"),
        ('{"id": 0, "db_id": "a", "predicted": null}', "predicted strings"),
    ],
)
def test_load_predictions_rejects_a_bad_line_and_names_it(
    tmp_path: Path, line: str, message: str
) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text('{"id": 0, "db_id": "a", "predicted": "x"}\n' + line + "\n")

    with pytest.raises(ValueError, match=rf"p\.jsonl:2: .*{message}"):
        load_predictions(path)


EXAMPLES = [
    Example(db_id="a", question="?", query="SELECT 1"),
    Example(db_id="b", question="?", query="SELECT 2"),
]


def test_align_orders_the_predictions_by_id() -> None:
    shuffled = [
        Prediction(id=1, db_id="b", predicted="two"),
        Prediction(id=0, db_id="a", predicted="one"),
    ]

    assert align(shuffled, EXAMPLES) == ["one", "two"]


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        ([0, 0, 1], "duplicate id 0"),
        ([0], "missing \\[1\\]"),
        ([0, 1, 2], "unknown \\[2\\]"),
    ],
)
def test_align_rejects_ids_that_do_not_match_the_questions(ids: list[int], message: str) -> None:
    db_ids = {0: "a", 1: "b", 2: "c"}
    predictions = [Prediction(id=i, db_id=db_ids[i], predicted="x") for i in ids]

    with pytest.raises(ValueError, match=message):
        align(predictions, EXAMPLES)


def test_align_rejects_a_prediction_for_another_database() -> None:
    predictions = [
        Prediction(id=0, db_id="a", predicted="x"),
        Prediction(id=1, db_id="a", predicted="x"),
    ]

    with pytest.raises(ValueError, match="id 1: db_id a is not b"):
        align(predictions, EXAMPLES)


def test_score_counts_the_questions_by_difficulty(
    spider_dir: Path, suite_dir: Path, predictions: list[Prediction]
) -> None:
    report = score(predictions, spider_dir, "dev", suite_dir)

    assert report["count"] == {"easy": 2, "medium": 1, "hard": 1, "extra": 2, "all": 6}


def test_score_execution_accuracy_uses_the_original_database(
    spider_dir: Path, suite_dir: Path, predictions: list[Prediction]
) -> None:
    report = score(predictions, spider_dir, "dev", suite_dir)

    # `SELECT 3` and `age > 40` happen to be right on the original database
    assert report["ex"] == {"easy": 1.0, "medium": 1.0, "hard": 1.0, "extra": 0.0, "all": 4 / 6}


def test_score_test_suite_accuracy_catches_right_by_luck(
    spider_dir: Path, suite_dir: Path, predictions: list[Prediction]
) -> None:
    report = score(predictions, spider_dir, "dev", suite_dir)

    assert report["ts"] == {"easy": 0.5, "medium": 1.0, "hard": 0.0, "extra": 0.0, "all": 2 / 6}


def test_score_exact_match(
    spider_dir: Path, suite_dir: Path, predictions: list[Prediction]
) -> None:
    report = score(predictions, spider_dir, "dev", suite_dir)

    assert report["em"] == {"easy": 0.5, "medium": 1.0, "hard": 0.0, "extra": 0.0, "all": 2 / 6}


def test_score_component_matching(
    spider_dir: Path, suite_dir: Path, predictions: list[Prediction]
) -> None:
    cm = score(predictions, spider_dir, "dev", suite_dir)["cm"]

    assert cm["medium"]["where"] == {"acc": 1.0, "rec": 1.0, "f1": 1.0}
    assert cm["medium"]["order"] == {"acc": 1.0, "rec": 1.0, "f1": 1.0}
    assert cm["hard"]["select"]["acc"] == 1.0  # both select `name`
    assert cm["hard"]["where"]["acc"] == 0.0  # `age > 40` is not `age > (SELECT ...)`


def test_score_without_the_test_suite_has_no_test_suite_accuracy(
    spider_dir: Path, predictions: list[Prediction]
) -> None:
    report = score(predictions, spider_dir, "dev")

    assert report["ts"] is None
    assert report["ex"]["all"] == 4 / 6
    assert report["em"]["all"] == 2 / 6


def test_score_fails_when_the_tokenizer_data_cannot_be_downloaded(
    spider_dir: Path, predictions: list[Prediction], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(score_module.nltk, "download", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="tokenizer data"):
        score(predictions, spider_dir, "dev")


def test_format_report_shows_a_dash_without_the_test_suite(
    spider_dir: Path, predictions: list[Prediction]
) -> None:
    lines = format_report(score(predictions, spider_dir, "dev")).splitlines()

    assert lines[0].split() == LEVELS
    assert [line.split()[0] for line in lines[1:]] == ["count", "EX", "TS", "EM"]
    assert lines[3].split() == ["TS", "-", "-", "-", "-", "-"]


def test_main_prints_the_report_and_writes_it_as_json(
    spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    argv = ["score", str(path), "--spider-dir", str(spider_dir), "--test-suite-dir", str(suite_dir)]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    assert "TS" in capsys.readouterr().out
    scores, metadata = read_scores(tmp_path)
    assert scores["ts"]["all"] == 2 / 6
    assert metadata["split"] == "dev"


def test_main_asks_to_download_a_missing_test_suite(
    spider_dir: Path,
    predictions: list[Prediction],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    argv = [
        "score",
        str(path),
        "--spider-dir",
        str(spider_dir),
        "--test-suite-dir",
        str(tmp_path / "none"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        main()

    assert "download_data.sh test-suite" in capsys.readouterr().err


def test_a_prediction_containing_value_is_not_rewritten(tmp_path: Path) -> None:
    spider = tmp_path / "spider"
    (spider / "database" / "shop").mkdir(parents=True)
    con = sqlite3.connect(spider / "database" / "shop" / "shop.sqlite")
    con.execute("CREATE TABLE item (item_id int PRIMARY KEY, value int)")
    con.execute("INSERT INTO item VALUES (1, 10), (2, 20)")
    con.commit()
    con.close()
    query = "SELECT max(value) FROM item"
    (spider / "dev.json").write_text(
        json.dumps([{"db_id": "shop", "question": "?", "query": query}])
    )
    (spider / "tables.json").write_text(
        json.dumps(
            [
                {
                    "db_id": "shop",
                    "table_names_original": ["item"],
                    "column_names_original": [[-1, "*"], [0, "item_id"], [0, "value"]],
                    "column_types": ["text", "number", "number"],
                    "primary_keys": [1],
                    "foreign_keys": [],
                }
            ]
        )
    )

    report = score([Prediction(id=0, db_id="shop", predicted=query)], spider, "dev")

    assert report["ex"]["all"] == 1.0
    assert report["em"]["all"] == 1.0


def make_test_split(root: Path, queries: list[str]) -> Path:
    """A Spider directory with only the test split: `test.json`, `test_tables.json`, `test_database/`."""
    make_database(root / "test_database" / "concert" / "concert.sqlite", extra_singer=False)
    test = [{"db_id": "concert", "question": "?", "query": query} for query in queries]
    (root / "test.json").write_text(json.dumps(test))
    (root / "test_tables.json").write_text(spider_tables := json.dumps(TABLES))
    assert spider_tables
    return root


def test_score_on_the_test_split_reads_the_test_files_and_has_no_test_suite(tmp_path: Path) -> None:
    spider = make_test_split(tmp_path, ["SELECT count(*) FROM singer", "SELECT name FROM singer"])
    predictions = [
        Prediction(id=0, db_id="concert", predicted="SELECT count(*) FROM singer"),
        Prediction(id=1, db_id="concert", predicted="SELECT age FROM singer"),
    ]

    report = score(predictions, spider, "test")

    assert report["count"]["all"] == 2
    assert report["ts"] is None
    assert report["ex"]["all"] == 0.5
    assert report["em"]["all"] == 0.5


def test_score_collapses_whitespace_in_the_gold_and_the_prediction(tmp_path: Path) -> None:
    # two Spider test gold queries contain a newline or a tab, which the official file format cannot hold
    spider = make_test_split(tmp_path, ["SELECT count(*)\n\tFROM singer"])
    predictions = [Prediction(id=0, db_id="concert", predicted="SELECT   count(*)\nFROM\tsinger\n")]

    report = score(predictions, spider, "test")

    assert report["ex"]["all"] == 1.0
    assert report["em"]["all"] == 1.0


def test_main_scores_the_test_split_without_a_test_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spider = make_test_split(tmp_path / "spider", ["SELECT count(*) FROM singer"])
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps({"id": 0, "db_id": "concert", "predicted": "SELECT count(*) FROM singer"}) + "\n"
    )
    argv = [
        "score",
        str(path),
        "--split",
        "test",
        "--spider-dir",
        str(spider),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()  # the test split needs no test suite

    assert "scores/run.report.json" in capsys.readouterr().out
    scores, metadata = read_scores(tmp_path)
    assert scores["ts"] is None
    assert metadata["split"] == "test"


def test_distinct_is_stripped_from_execution_unless_kept(tmp_path: Path) -> None:
    spider = make_test_split(tmp_path, ["SELECT count(DISTINCT name) FROM singer"])
    con = sqlite3.connect(spider / "test_database" / "concert" / "concert.sqlite")
    con.execute("INSERT INTO singer VALUES (4, 'Tim', 33)")  # two Tims: 3 distinct names, 4 names
    con.commit()
    con.close()
    predictions = [Prediction(id=0, db_id="concert", predicted="SELECT count(name) FROM singer")]

    stripped = score(predictions, spider, "test")
    kept = score(predictions, spider, "test", keep_distinct=True)

    assert (stripped["ex"]["all"], kept["ex"]["all"]) == (1.0, 0.0)
    assert stripped["em"]["all"] == kept["em"]["all"] == 1.0  # EM ignores DISTINCT either way


def test_main_passes_keep_distinct_on_and_notes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spider = make_test_split(tmp_path / "spider", ["SELECT count(*) FROM singer"])
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps({"id": 0, "db_id": "concert", "predicted": "SELECT count(*) FROM singer"}) + "\n"
    )
    argv = ["score", str(path), "--split", "test", "--spider-dir", str(spider), "--keep-distinct"]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    assert "DISTINCT kept" in capsys.readouterr().out
    assert read_scores(tmp_path)[1]["keep_distinct"] is True


def test_one_line_keeps_the_spaces_inside_quotes() -> None:
    sql = "SELECT  a\n FROM\tt WHERE b  =  'x  y' AND c = \"p\tq\"  AND d = 'it''s  ok'"

    assert one_line(sql) == "SELECT a FROM t WHERE b = 'x  y' AND c = \"p q\" AND d = 'it''s  ok'"


def test_a_value_with_repeated_spaces_is_matched_as_it_is(tmp_path: Path) -> None:
    gold = "SELECT name FROM singer WHERE name = 'Tim  Lee'"
    spider = make_test_split(tmp_path, [gold])
    con = sqlite3.connect(spider / "test_database" / "concert" / "concert.sqlite")
    con.execute("INSERT INTO singer VALUES (5, 'Tim  Lee', 30)")
    con.commit()
    con.close()
    right = [Prediction(id=0, db_id="concert", predicted=gold)]
    # empty like the gold would be if its two spaces were collapsed
    wrong = [Prediction(id=0, db_id="concert", predicted="SELECT name FROM singer WHERE age = 99")]

    assert score(right, spider, "test")["ex"]["all"] == 1.0
    assert score(wrong, spider, "test")["ex"]["all"] == 0.0


@pytest.mark.parametrize(
    ("reply", "sql"),
    [
        ("SELECT count(*) FROM singer", "SELECT count(*) FROM singer"),
        ("SELECT count(*) FROM singer;", "SELECT count(*) FROM singer"),
        (
            "To count them, use this query:\n\n```sql\nSELECT COUNT(*)\nFROM singer;\n```\n\nIt counts the rows.",
            "SELECT COUNT(*)\nFROM singer",
        ),
        ("```\nSELECT name FROM singer\n```", "SELECT name FROM singer"),
        ("```SQL\nSELECT name FROM singer\n```", "SELECT name FROM singer"),
        ("```SELECT name FROM singer```", "SELECT name FROM singer"),
        (
            "```SELECT\nname FROM singer```",
            "SELECT\nname FROM singer",
        ),  # SELECT is not a language tag
        ("```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```", "SELECT 1"),  # the first block
        ("Use this:\n```sql\nSELECT name FROM singer", "SELECT name FROM singer"),  # never closed
        ("SELECT name FROM singer; This lists the singers.", "SELECT name FROM singer"),
        (
            "SELECT name FROM singer WHERE name = 'a;b'; DROP TABLE x",
            "SELECT name FROM singer WHERE name = 'a;b'",
        ),
        (
            'SELECT name FROM singer WHERE name = "a;b"',
            'SELECT name FROM singer WHERE name = "a;b"',
        ),
        (
            "Jetblue Airways is affiliated with the United States.",
            "Jetblue Airways is affiliated with the United States.",
        ),
        ("", ""),
        ("  \n ", ""),
    ],
)
def test_extract_sql(reply: str, sql: str) -> None:
    assert extract_sql(reply) == sql


def fenced(gold: str) -> str:
    return (
        f"To answer the question, use this query:\n\n```sql\n{gold};\n```\n\nIt returns the rows."
    )


def test_score_takes_the_sql_out_of_the_replies_by_default(
    spider_dir: Path, suite_dir: Path
) -> None:
    predictions = [
        Prediction(id=i, db_id="concert", predicted=fenced(gold))
        for i, (gold, _, _) in enumerate(QUESTIONS)
    ]

    report = score(predictions, spider_dir, "dev", suite_dir)

    assert report["ex"]["all"] == report["ts"]["all"] == report["em"]["all"] == 1.0  # type: ignore[index]


def test_score_can_score_the_replies_as_written(spider_dir: Path, suite_dir: Path) -> None:
    predictions = [
        Prediction(id=i, db_id="concert", predicted=fenced(gold))
        for i, (gold, _, _) in enumerate(QUESTIONS)
    ]

    report = score(predictions, spider_dir, "dev", suite_dir, extract=False)

    assert report["ex"]["all"] == report["em"]["all"] == 0.0  # the fence is not SQL


def test_main_raw_scores_the_replies_as_written(
    spider_dir: Path,
    suite_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": i, "db_id": "concert", "predicted": fenced(gold)}) + "\n"
            for i, (gold, _, _) in enumerate(QUESTIONS)
        )
    )
    argv = ["score", str(path), "--spider-dir", str(spider_dir), "--test-suite-dir", str(suite_dir)]
    monkeypatch.setattr(sys, "argv", argv)

    main()
    assert "scored as written" not in capsys.readouterr().out
    first, first_metadata = read_scores(tmp_path)
    assert first["ex"]["all"] == 1.0
    assert first_metadata["predictions_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(sys, "argv", [*argv, "--raw"])
    main()
    assert "scored as written" in capsys.readouterr().out
    scores, metadata = read_scores(tmp_path)
    assert metadata["extraction"] is False
    assert scores["ex"]["all"] == 0.0


def predictions_file(
    tmp_path: Path, predictions: list[Prediction], metadata: dict[str, object] | None
) -> Path:
    """Writes `run.jsonl` and, if given, `run.meta.json` with the hash of the predictions."""
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    if metadata is not None:
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        (tmp_path / "run.meta.json").write_text(
            json.dumps({"predictions_sha256": sha256, **metadata})
        )
    return path


METADATA = {
    "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "model_revision": "abc123",
    "run": "baseline",
    "split": "dev",
    "generation": {"do_sample": False, "max_new_tokens": 256, "batch_size": 16},
}


def run_main(
    path: Path, spider_dir: Path, suite_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    argv = ["score", str(path), "--spider-dir", str(spider_dir), "--test-suite-dir", str(suite_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    scores = json.loads((Path("scores") / f"{path.stem}.report.json").read_text())
    metadata = json.loads((Path("scores") / f"{path.stem}.report.meta.json").read_text())
    return {**scores, **metadata}


def test_main_reads_the_metadata_next_to_the_predictions(
    tmp_path: Path,
    spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = predictions_file(tmp_path, predictions, METADATA)

    report = run_main(path, spider_dir, suite_dir, monkeypatch)

    assert (report["model"], report["run"], report["model_revision"]) == (
        METADATA["model"],
        "baseline",
        "abc123",
    )


def test_main_scores_without_metadata_and_leaves_the_fields_empty(
    tmp_path: Path,
    spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = predictions_file(tmp_path, predictions, None)

    report = run_main(path, spider_dir, suite_dir, monkeypatch)

    assert report["model"] is None


def test_main_stops_when_the_metadata_belongs_to_another_predictions_file(
    tmp_path: Path,
    spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = predictions_file(tmp_path, predictions, METADATA)
    meta = json.loads((tmp_path / "run.meta.json").read_text())
    (tmp_path / "run.meta.json").write_text(json.dumps({**meta, "predictions_sha256": "0" * 64}))

    with pytest.raises(SystemExit):
        run_main(path, spider_dir, suite_dir, monkeypatch)

    assert "describes another predictions file" in capsys.readouterr().err
    assert not (tmp_path / "scores").exists()


def test_main_accepts_metadata_without_a_hash(
    tmp_path: Path,
    spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = predictions_file(tmp_path, predictions, METADATA)
    (tmp_path / "run.meta.json").write_text(json.dumps(METADATA))  # no predictions_sha256

    assert run_main(path, spider_dir, suite_dir, monkeypatch)["run"] == "baseline"


def test_main_warns_when_the_metadata_is_for_another_split(
    tmp_path: Path,
    spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = predictions_file(tmp_path, predictions, {**METADATA, "split": "test"})

    run_main(path, spider_dir, suite_dir, monkeypatch)

    assert "metadata is for the test split, scoring dev" in capsys.readouterr().err


def test_write_report_writes_only_the_scores(
    spider_dir: Path, suite_dir: Path, predictions: list[Prediction], tmp_path: Path
) -> None:
    report = score(predictions, spider_dir, "dev", suite_dir)

    write_report(report, tmp_path / "out" / "x.report.json")

    assert json.loads((tmp_path / "out" / "x.report.json").read_text()) == report
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["x.report.json"]


def test_main_writes_the_metadata_next_to_the_scores_and_keeps_them_apart(
    tmp_path: Path,
    spider_dir: Path,
    suite_dir: Path,
    predictions: list[Prediction],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = predictions_file(tmp_path, predictions, None)
    run_main(path, spider_dir, suite_dir, monkeypatch)

    scores, metadata = read_scores(tmp_path)

    assert set(scores) == {"count", "ex", "ts", "em", "cm"}
    assert not set(scores) & set(metadata)
    assert metadata["predictions_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
