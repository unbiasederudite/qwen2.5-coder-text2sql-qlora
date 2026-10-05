# spider_evaluation

Official Spider evaluation code, vendored with small changes.

| File | Description |
| --- | --- |
| `evaluation.py` | Scoring script. |
| `exec_eval.py` | Execution comparison. |
| `parse.py` | `DISTINCT` removal and value plugging. |
| `process_sql.py` | SQL parser. |
| `LICENSE` | Apache-2.0, as upstream. |

## Source

[taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) at commit `e97acc546ecbee8fa27fa8dbf025ef61493a876c` (2022-08-02).

## Changes

Each change is marked `# Modified` in the code:

- `evaluation.py`: relative import.
- `exec_eval.py`: relative import.
- `exec_eval.py`: the database is opened read-only, and a missing file is an error instead of a new empty database.
- `exec_eval.py`: sqlite itself stops the query at the deadline, instead of `asyncio.wait_for`.
- `evaluation.py`: `evaluate` returns the scores.
- `evaluation.py`: a prediction containing `value` is no longer rewritten to `1`.

`parse.py`, `process_sql.py` and `LICENSE` are byte-identical to upstream.
