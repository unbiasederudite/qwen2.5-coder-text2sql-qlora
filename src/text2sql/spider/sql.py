"""Plain-text handling of SQL: normalizing queries and extracting them from model replies."""

import re

# The first markdown code block: an optional language tag (not SELECT or WITH), then the code
_FENCE = re.compile(
    r"```[ \t]*(?:sql\b[ \t]*|(?!(?:select|with)\b)[\w+-]+[ \t]*\n)?(.*?)(?:```|\Z)",
    re.DOTALL | re.IGNORECASE,
)


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
    match = _FENCE.search(reply)
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
