"""The predictions file of a Spider split: one JSON object per line."""

import json
from pathlib import Path
from typing import IO, TypedDict


class Prediction(TypedDict):
    """One line of a predictions file."""

    id: int  # position of the question in the split
    db_id: str  # database name
    predicted: str  # model output


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


def append_prediction(file: IO[str], prediction: Prediction) -> None:
    """Writes one prediction as a line.

    Args:
        file (IO[str]): Predictions file opened for appending.
        prediction (Prediction): Prediction to write.
    """
    file.write(json.dumps(prediction) + "\n")
