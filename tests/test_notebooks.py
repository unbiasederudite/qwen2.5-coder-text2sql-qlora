"""Hygiene checks of the notebooks: valid cells, Colab cell titles and no stored run output."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

NOTEBOOKS = sorted(
    path
    for path in (Path(__file__).parent.parent / "notebooks").glob("*.ipynb")
    if not path.name.endswith("_output.ipynb")  # written by `colab exec` and ignored by git
)
COLAB_NOTEBOOKS = [path for path in NOTEBOOKS if path.name.startswith("colab_")]
TITLE = re.compile(r"^\s*#\s*@title\s+\S", re.MULTILINE)  # as the Colab CLI reads it


def code_cells(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Returns the index and the content of every code cell of a notebook."""
    cells = json.loads(path.read_text())["cells"]
    return [(i, cell) for i, cell in enumerate(cells) if cell["cell_type"] == "code"]


def source(cell: dict[str, Any]) -> str:
    return "".join(cell["source"])


def test_notebooks_are_found() -> None:
    assert NOTEBOOKS
    assert COLAB_NOTEBOOKS


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_has_no_stored_output(path: Path) -> None:
    dirty = [
        i for i, cell in code_cells(path) if cell["outputs"] or cell["execution_count"] is not None
    ]

    assert not dirty, f"code cells with stored output: {dirty}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_code_cells_compile(path: Path) -> None:
    for i, cell in code_cells(path):
        lines = source(cell).splitlines()
        if lines and lines[0].startswith("%%"):  # a cell magic, not Python
            continue
        python = "\n".join(line for line in lines if not line.startswith(("%", "!")))
        compile(python, f"{path.name} cell {i}", "exec")


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda path: path.name)
def test_colab_notebook_code_cells_have_a_title(path: Path) -> None:
    untitled = [i for i, cell in code_cells(path) if not TITLE.search(source(cell))]

    assert not untitled, f"code cells without '# @title': {untitled}"


def test_colab_notebooks_share_the_setup_cell() -> None:
    setup_cells = {
        path.name: next(
            source(cell)
            for _, cell in code_cells(path)
            if "# @title Install the project" in source(cell)
        )
        for path in COLAB_NOTEBOOKS
    }

    assert len(set(setup_cells.values())) == 1, f"setup cells differ: {sorted(setup_cells)}"
