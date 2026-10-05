# qwen2.5-coder-text2sql-qlora

## Models

QLoRA with 4-bit models.

| Model | GPU memory (4-bit) |
| --- | --- |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.2 GB |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | 2.1 GB |

## Data

Download from the project root. At least one target is required:

```bash
bash scripts/download_data.sh spider test-suite
```

- `spider`: the official [Spider](https://yale-lily.github.io/spider) release.
- `test-suite`: extra versions of the dev databases for scoring (1.3 GB download, 4.9 GB on disk).

## Scoring

Score a predictions file from the project root:

```bash
uv run python -m text2sql.score predictions/<name>.jsonl
```

Scoring is built around the vendored official [test-suite evaluation](https://github.com/taoyds/test-suite-sql-eval), which provides these metrics, each by difficulty (easy, medium, hard, extra) and over all questions:

| Metric | Name | Description |
| --- | --- | --- |
| EX | Execution accuracy | The predicted query returns the same rows as the gold query on the original database. |
| TS | Test-suite accuracy | The same check on every extra version of the database, so a lucky answer fails. Dev only. |
| EM | Exact set match | The predicted query has the same parts as the gold query, such as select and where. Values are ignored. |
| CM | Component matching | How well each part of the query matches, scored separately for select, where, order and so on. |

## Training sample

Each sample is three chat messages:

- `system`: the database schema as `CREATE TABLE` statements
- `user`: the question
- `assistant`: the gold SQL

Example:

```
system:
    CREATE TABLE "stadium" (
    "Stadium_ID" int,
    "Location" text,
    "Name" text,
    "Capacity" int,
    "Highest" int,
    "Lowest" int,
    "Average" int,
    PRIMARY KEY ("Stadium_ID")
    );

    CREATE TABLE "singer" (
    "Singer_ID" int,
    "Name" text,
    "Country" text,
    "Song_Name" text,
    "Song_release_year" text,
    "Age" int,
    "Is_male" bool,
    PRIMARY KEY ("Singer_ID")
    );

    CREATE TABLE "concert" (
    "concert_ID" int,
    "concert_Name" text,
    "Theme" text,
    "Stadium_ID" text,
    "Year" text,
    PRIMARY KEY ("concert_ID"),
    FOREIGN KEY ("Stadium_ID") REFERENCES "stadium"("Stadium_ID")
    );

    CREATE TABLE "singer_in_concert" (
    "concert_ID" int,
    "Singer_ID" text,
    PRIMARY KEY ("concert_ID","Singer_ID"),
    FOREIGN KEY ("concert_ID") REFERENCES "concert"("concert_ID"),
    FOREIGN KEY ("Singer_ID") REFERENCES "singer"("Singer_ID")
    );

user:
    How many singers do we have?

assistant:
    SELECT count(*) FROM singer
```

## Colab

The GPU work (models, baseline, training) runs on a free Colab T4. Everything else runs locally.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unbiasederudite/qwen2.5-coder-text2sql-qlora/blob/main/notebooks/colab_model_check.ipynb)

The notebook's first cell sets `REPO` and `REF`. The setup script reads them to clone the repo and install it, and the cell then downloads the data.

### VS Code

If VS Code did not recommend the Colab extension, or it is not installed yet:

```bash
code --install-extension google.colab
```
