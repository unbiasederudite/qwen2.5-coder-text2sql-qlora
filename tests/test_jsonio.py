"""Tests for text2sql.jsonio."""

import json
from pathlib import Path

from text2sql import jsonio


def test_write_json_writes_lists_too(tmp_path: Path) -> None:
    jsonio.write_json(tmp_path / "log.json", [{"step": 25}])

    assert json.loads((tmp_path / "log.json").read_text()) == [{"step": 25}]
