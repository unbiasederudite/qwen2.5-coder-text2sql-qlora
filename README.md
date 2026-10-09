# qwen2.5-coder-text2sql-qlora

[![CI](https://github.com/unbiasederudite/qwen2.5-coder-text2sql-qlora/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/unbiasederudite/qwen2.5-coder-text2sql-qlora/actions/workflows/ci.yml)

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

The data goes to `data/spider_data/` and `data/test_suite_data/`. See [Reproduce the results](#reproduce-the-results) for the full sequence.

## Scoring

Score a predictions file from the project root:

```bash
uv run python -m text2sql.score predictions/<name>.jsonl
```

Only the predictions file is required:

| Parameter | Values | Description |
| --- | --- | --- |
| `predictions` | path to a `.jsonl` file | Predictions to score. |
| `--split` | `dev` (default), `test` | Spider split the predictions are for. |
| `--spider-dir` | directory, default `data/spider_data` | Spider release. |
| `--test-suite-dir` | directory, default `data/test_suite_data` | Test-suite databases for TS, used for `dev` only. |
| `--keep-distinct` | flag | Keep `DISTINCT` in EX and TS, the official script strips it. EM and CM always ignore it. |
| `--raw` | flag | Score the replies as written, without taking the SQL out of code blocks. |

The scorer prints the scores by difficulty and writes `<name>.report.json` and `<name>.report.meta.json` to `scores/`. See [Reproduce the results](#reproduce-the-results) for the full sequence.

Scoring is built around the vendored official [test-suite evaluation](https://github.com/taoyds/test-suite-sql-eval), which provides these metrics, each by difficulty (easy, medium, hard, extra) and over all questions:

| Metric | Name | Description |
| --- | --- | --- |
| EX | Execution accuracy | The predicted query returns the same rows as the gold query on the original database. |
| TS | Test-suite accuracy | The same check on every extra version of the database, so a lucky answer fails. Dev only. |
| EM | Exact set match | The predicted query has the same parts as the gold query, such as select and where. Values are ignored. |
| CM | Component matching | How well each part of the query matches, scored separately for select, where, order and so on. |

Only text-to-SQL is evaluated. The fine-tuned models are intended for this task only, and any loss of general capabilities is possible and not measured.

## Baseline results

Scores of the models on the Spider dev split before fine-tuning:

| Model | EX | TS | EM |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 0.652 | 0.555 | 0.428 |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | 0.712 | 0.627 | 0.457 |

The scores and their metadata are in `scores/`, the predictions in `predictions/`. See [Reproduce the results](#reproduce-the-results) to run them again.

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

## Fine-tuning

QLoRA fine-tuning of one model on the Spider training data, `train_spider` and `train_others`, without repeated questions and gold queries that fail to run. That is 8,647 samples, and the 82 that are longer than 2,048 tokens are dropped. The loss is computed on the SQL only. Dev and test are not used for training.

The LoRA and training settings are common QLoRA defaults and were not tuned:

| Setting | Value |
| --- | --- |
| LoRA | rank 16, alpha 32, dropout 0.05, on all linear layers |
| Epochs | 2 |
| Learning rate | `2e-4`, cosine schedule, 50 warmup steps |
| Batch | 4 samples per step, 4 steps added up per update (16 samples) |
| Optimizer | AdamW, 8-bit and paged |
| Precision | fp16, with gradient checkpointing |
| Seed | 42 |
| Checkpoints | after every epoch, only the last one is kept; a rerun resumes from it |

The settings are in `configs/qlora.yaml`.

## Colab

The GPU work (generation, training) runs on a free Colab T4. Everything else runs locally.

| Notebook | Description | Open |
| --- | --- | --- |
| `colab_generate_predictions.ipynb` | Predictions of one model on one split. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unbiasederudite/qwen2.5-coder-text2sql-qlora/blob/main/notebooks/colab_generate_predictions.ipynb) |
| `colab_fine_tune.ipynb` | QLoRA fine-tuning of one model on Spider. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unbiasederudite/qwen2.5-coder-text2sql-qlora/blob/main/notebooks/colab_fine_tune.ipynb) |

### colab_generate_predictions

Parameters, all required:

| Parameter | Values | Description |
| --- | --- | --- |
| `MODEL` | `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-3B-Instruct` | Model that answers the questions. |
| `SPLIT` | `dev`, `test` | Spider split to answer. |
| `RUN` | any name | Name of the run, part of the output file name. |
| `BATCH_SIZE` | integer, for example `16` | Questions generated at once. |

Pass them with the [Colab CLI](https://github.com/googlecolab/google-colab-cli) as `--env` options:

```bash
uv run colab exec -s <session> -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=<model> --env SPLIT=<split> --env RUN=<run> --env BATCH_SIZE=<batch size>
```

In the browser or VS Code, uncomment the `%env` lines at the top of the notebook's first cell and set the values. Keep them commented when you use the CLI, as they would override its values.

The notebook writes `<model>_<run>_<split>.jsonl` and `<model>_<run>_<split>.meta.json` to `/content/qwen2.5-coder-text2sql-qlora/predictions_colab/` on the session. Each batch is written as it is done, so running the same command again continues where a stopped run ended, and does nothing if every question is answered. To generate predictions again, restart the session. See [Reproduce the results](#reproduce-the-results) for the full sequence.

### colab_fine_tune

Parameters, all required:

| Parameter | Values | Description |
| --- | --- | --- |
| `MODEL` | `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-3B-Instruct` | Model to fine-tune. |
| `CONFIG` | path, for example `configs/qlora.yaml` | Training config: the splits to train on, and the LoRA and training settings. |

```bash
uv run colab exec -s <session> -f notebooks/colab_fine_tune.ipynb --timeout 28800 \
  --env MODEL=<model> --env CONFIG=<config>
```

In the browser or VS Code, uncomment the `%env` lines at the top of the notebook's first cell and set the values. Keep them commented when you use the CLI, as they would override its values.

The notebook writes the adapter, `training.meta.json` and `training.log.json` to `/content/qwen2.5-coder-text2sql-qlora/adapters_colab/<model>/` on the session. A checkpoint is saved after every epoch, so running the same command again continues from the last one, and does nothing if the adapter is finished. To train again, restart the session. See [Reproduce the results](#reproduce-the-results) for the full sequence.

### VS Code

If VS Code did not recommend the Colab extension, or it is not installed yet:

```bash
code --install-extension google.colab
```

## Checks

Lint, formatting, types and tests, the same steps CI runs on every push:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

## Reproduce the results

Run every command from the project root, in order. All Colab steps use one session, `qlora`.

### 1. Setup

Install the project, and download the data for scoring:

```bash
uv sync
bash scripts/download_data.sh spider test-suite
```

The data is in `data/spider_data/` and `data/test_suite_data/`.

### 2. Start the Colab session

The first call opens a browser to sign in:

```bash
uv run colab new -s qlora --gpu T4
```

### 3. Evaluate the baseline models

The same notebook runs for each model:

```bash
uv run colab exec -s qlora -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct --env SPLIT=dev --env RUN=baseline --env BATCH_SIZE=16
uv run colab exec -s qlora -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=Qwen/Qwen2.5-Coder-3B-Instruct --env SPLIT=dev --env RUN=baseline --env BATCH_SIZE=16
```

The predictions are in `/content/qwen2.5-coder-text2sql-qlora/predictions_colab/` on the session.

### 4. Download the predictions

Each run writes its predictions and their metadata. Files on the session are lost when it stops:

```bash
mkdir -p predictions
for name in qwen2.5-coder-1.5b-instruct_baseline_dev qwen2.5-coder-3b-instruct_baseline_dev; do
  for ext in jsonl meta.json; do
    uv run colab download -s qlora /content/qwen2.5-coder-text2sql-qlora/predictions_colab/$name.$ext predictions/$name.$ext
  done
done
```

The files are in `predictions/`.

### 5. Score the baseline predictions

```bash
uv run python -m text2sql.score predictions/qwen2.5-coder-1.5b-instruct_baseline_dev.jsonl
uv run python -m text2sql.score predictions/qwen2.5-coder-3b-instruct_baseline_dev.jsonl
```

The scores are in `scores/`.

### 6. Fine-tune the models

The same notebook runs for each model:

```bash
uv run colab exec -s qlora -f notebooks/colab_fine_tune.ipynb --timeout 28800 \
  --env MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct --env CONFIG=configs/qlora.yaml
uv run colab exec -s qlora -f notebooks/colab_fine_tune.ipynb --timeout 28800 \
  --env MODEL=Qwen/Qwen2.5-Coder-3B-Instruct --env CONFIG=configs/qlora.yaml
```

The adapters are in `/content/qwen2.5-coder-text2sql-qlora/adapters_colab/` on the session.

### 7. Download the adapters

Each run writes its adapter, a directory of several files, with the metadata and the log of the training. Files on the session are lost when it stops:

```bash
for name in qwen2.5-coder-1.5b-instruct qwen2.5-coder-3b-instruct; do
  mkdir -p adapters/$name
  for file in adapter_model.safetensors adapter_config.json README.md tokenizer.json tokenizer_config.json chat_template.jinja training_args.bin training.meta.json training.log.json; do
    uv run colab download -s qlora /content/qwen2.5-coder-text2sql-qlora/adapters_colab/$name/$file adapters/$name/$file
  done
done
```

The files are in `adapters/`.

### 8. Close the session

```bash
uv run colab stop -s qlora
```
