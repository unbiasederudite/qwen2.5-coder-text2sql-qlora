from text2sql.data import Sample
from text2sql.prompt import build_completion, build_prompt

SAMPLE = Sample(
    db_id="concert",
    question="How many singers are there?",
    query="SELECT count(*) FROM singer",
    schema=["CREATE TABLE singer (name TEXT);", "CREATE TABLE concert (year INT);"],
)


def test_prompt_is_schema_then_question() -> None:
    assert build_prompt(SAMPLE) == [
        {
            "role": "system",
            "content": "CREATE TABLE singer (name TEXT);\n\nCREATE TABLE concert (year INT);",
        },
        {"role": "user", "content": "How many singers are there?"},
    ]


def test_completion_is_gold_sql() -> None:
    assert build_completion(SAMPLE) == [
        {"role": "assistant", "content": "SELECT count(*) FROM singer"}
    ]
