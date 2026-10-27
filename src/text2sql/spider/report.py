"""The scores report: its shape and its table."""

from typing import TypedDict

LEVELS = ("easy", "medium", "hard", "extra", "all")


class Report(TypedDict):
    """How a predictions file scored, by difficulty level and over all questions."""

    count: dict[str, int]  # questions per level
    ex: dict[str, float]  # execution accuracy on the original database
    ts: dict[str, float] | None  # test-suite accuracy, None without the test suite
    em: dict[str, float]  # exact set match
    cm: dict[str, dict[str, dict[str, float]]]  # level -> component -> acc, rec, f1


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
