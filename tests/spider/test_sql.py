"""Tests for text2sql.spider.sql."""

import pytest

from text2sql.spider.sql import extract_sql, one_line


def test_one_line_keeps_the_spaces_inside_quotes() -> None:
    sql = "SELECT  a\n FROM\tt WHERE b  =  'x  y' AND c = \"p\tq\"  AND d = 'it''s  ok'"

    assert one_line(sql) == "SELECT a FROM t WHERE b = 'x  y' AND c = \"p q\" AND d = 'it''s  ok'"


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
