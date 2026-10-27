"""Writing JSON files."""

import json
from pathlib import Path


def _to_json(value: object) -> object:
    """Serializes an object that `json` cannot.

    Args:
        value (object): Object to serialize.

    Returns:
        object: Its `to_dict()`, or its string.
    """
    return value.to_dict() if hasattr(value, "to_dict") else str(value)


def write_json(path: Path, data: object) -> None:
    """Writes data as indented JSON.

    Args:
        path (Path): File to write.
        data (object): Data to write.
    """
    path.write_text(json.dumps(data, indent=2, default=_to_json) + "\n")
