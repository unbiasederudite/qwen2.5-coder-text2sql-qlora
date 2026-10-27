# spider

The Spider dataset: files, databases, samples, predictions and scoring.

| Path | Description |
| --- | --- |
| `dataset.py` | Spider files: splits, questions and schemas. |
| `database.py` | Spider's SQLite databases. |
| `samples.py` | Text-to-SQL samples built from Spider. |
| `predictions.py` | The predictions file of a Spider split: one JSON object per line. |
| `sql.py` | Plain-text handling of SQL: normalizing queries and extracting them from model replies. |
| `scoring.py` | Scoring of predictions with the official Spider evaluation code. |
| `report.py` | The scores report: its shape and its table. |
| `evaluation/` | The official Spider evaluation code, vendored from taoyds/test-suite-sql-eval. |
