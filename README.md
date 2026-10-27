# qwen2.5-coder-text2sql-qlora

[![CI](https://github.com/unbiasederudite/qwen2.5-coder-text2sql-qlora/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/unbiasederudite/qwen2.5-coder-text2sql-qlora/actions/workflows/ci.yml)

## Models

QLoRA with 4-bit models. The supported models, the 4-bit and the generation settings are in `configs/model.yaml`.

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

## Results

Scores of the models on the Spider dev split (1,034 questions) and test split (2,147 questions), before fine-tuning (baseline) and after it (fine-tuned, with the [adapters](#adapters)).

### Dev

| Model | Variant | EX | TS | EM |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | baseline | 0.652 | 0.555 | 0.428 |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | fine-tuned | 0.734 | 0.677 | 0.668 |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | baseline | 0.712 | 0.627 | 0.457 |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | fine-tuned | 0.778 | 0.722 | 0.735 |

### Test

The test split has no test-suite databases, so there is no TS:

| Model | Variant | EX | EM |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | baseline | 0.645 | 0.413 |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | fine-tuned | 0.732 | 0.662 |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | baseline | 0.715 | 0.425 |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | fine-tuned | 0.795 | 0.709 |

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

The settings are in `configs/train.yaml`.

## Adapters

The fine-tuned LoRA adapters are on Hugging Face and not in git, `adapters/` is ignored:

| Model | Adapter |
| --- | --- |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | [unbiasederudite/qwen2.5-coder-1.5b-text2sql-qlora](https://huggingface.co/unbiasederudite/qwen2.5-coder-1.5b-text2sql-qlora) |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | [unbiasederudite/qwen2.5-coder-3b-text2sql-qlora](https://huggingface.co/unbiasederudite/qwen2.5-coder-3b-text2sql-qlora) |

Each repository holds the adapter weights and config, the tokenizer files, `training.meta.json` and `training.log.json`. An adapter goes on top of its own base model only, and expects the chat format of the [training sample](#training-sample).

### Download

```bash
uv run hf download unbiasederudite/qwen2.5-coder-1.5b-text2sql-qlora --local-dir adapters/qwen2.5-coder-1.5b-instruct
uv run hf download unbiasederudite/qwen2.5-coder-3b-text2sql-qlora --local-dir adapters/qwen2.5-coder-3b-instruct
```

### Use

With this project, pass the repository as `ADAPTER` to [`colab_generate_predictions`](#colab_generate_predictions), or as the third argument of `load_model`. It needs a CUDA GPU and `uv sync --group colab`:

```python
from text2sql.config import load_model_config
from text2sql.model import load_model

model, tokenizer = load_model(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    load_model_config(),
    "unbiasederudite/qwen2.5-coder-1.5b-text2sql-qlora",
)
```

Without this project, with [PEFT](https://github.com/huggingface/peft):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(base),
    "unbiasederudite/qwen2.5-coder-1.5b-text2sql-qlora",
)
tokenizer = AutoTokenizer.from_pretrained(base)
```

## Metadata

Every run writes a `.meta.json` file that says what produced its output, so a result can be traced back to its code, data, model and settings:

| File | Written by | Describes |
| --- | --- | --- |
| `predictions/<name>.meta.json` | `colab_generate_predictions` | One generation run. |
| `scores/<name>.report.meta.json` | the scoring command | One scoring run. |
| `adapters/<model>/training.meta.json` | `colab_fine_tune` | One training run. |

**Generation**

- `run`, `model`, `model_revision`, `quantization`: the run name, the base model with its Hugging Face commit, and its quantization settings.
- `adapter` and `adapter_revision`: only for fine-tuned runs, the adapter repository and the commit that was loaded. With a local adapter directory, `adapter_sha256` holds the hash of its weights instead.
- `tokenizer`, `generation`: the padding side, and the generation settings with the `batch_size`.
- `split`, `split_file_sha256`, `questions`, `already_done`: the Spider split and the hash of its file, how many questions it has, and how many predictions were already in the file when the run started.
- `predictions_sha256`: the hash of the predictions file.
- `started_at`, `finished_at`, `duration_seconds`: when the run ran.
- `repo`, `commit`, `python`, `packages`, `gpu`: the code, the Python and package versions, and the GPU.

**Scoring**

- `run`, `model`, `model_revision`, `adapter`, `adapter_revision`: copied from the generation metadata next to the predictions file.
- `split`, `predictions_sha256`: the split that was scored, and the hash of the predictions file.
- `keep_distinct`, `extraction`: the scoring options, `--keep-distinct` and whether the SQL was taken out of the replies (off with `--raw`).
- `repo`, `commit`, `sqlite_version`: the code, and the SQLite that ran the queries.

The scorer stops if the generation metadata describes another predictions file, and warns if it is for another split. Without it, `run`, `model` and `model_revision` stay empty.

**Training**

- `model`, `model_revision`, `quantization`: as for generation.
- `data`, `lora`, `training`: the training samples (`splits`, `clean`, `max_length`), the LoRA settings and the trainer settings.
- `split_files_sha256`, `samples`, `dropped_too_long`: the hashes of the training split files, how many samples were trained on, and how many were dropped for being too long.
- `resumed`: whether the run continued from a checkpoint.
- `started_at`, `finished_at`, `duration_seconds`, `repo`, `commit`, `python`, `packages`, `gpu`: as for generation.

A field that could not be read, such as `gpu` without a GPU, is written as `null`, and the run prints a warning.

## Colab

The GPU work (generation, training) runs on a free Colab T4. Everything else runs locally.

| Notebook | Description | Open |
| --- | --- | --- |
| `colab_generate_predictions.ipynb` | Predictions of one model on one split. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unbiasederudite/qwen2.5-coder-text2sql-qlora/blob/main/notebooks/colab_generate_predictions.ipynb) |
| `colab_fine_tune.ipynb` | QLoRA fine-tuning of one model on Spider. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unbiasederudite/qwen2.5-coder-text2sql-qlora/blob/main/notebooks/colab_fine_tune.ipynb) |

### colab_generate_predictions

Parameters:

| Parameter | Values | Description |
| --- | --- | --- |
| `MODEL` | `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-3B-Instruct` | Model that answers the questions. Required. |
| `SPLIT` | `dev`, `test` | Spider split to answer. Required. |
| `RUN` | any name | Name of the run, part of the output file name. Required. |
| `BATCH_SIZE` | integer, for example `16` | Questions generated at once. Required. |
| `ADAPTER` | Hugging Face repository, for example `<user>/<adapter>` | LoRA adapter put on the model. Optional, without it the model is the baseline. |

Pass them with the [Colab CLI](https://github.com/googlecolab/google-colab-cli) as `--env` options:

```bash
uv run colab exec -s <session> -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=<model> --env SPLIT=<split> --env RUN=<run> --env BATCH_SIZE=<batch size> \
  [--env ADAPTER=<adapter>]
```

In the browser or VS Code, uncomment the `%env` lines at the top of the notebook's first cell and set the values. Keep them commented when you use the CLI, as they would override its values.

The notebook writes `<model>_<run>_<split>.jsonl` and `<model>_<run>_<split>.meta.json` to `/content/qwen2.5-coder-text2sql-qlora/predictions_colab/` on the session. Each batch is written as it is done, so running the same command again continues where a stopped run ended, and does nothing if every question is answered. To generate predictions again, restart the session. See [Reproduce the results](#reproduce-the-results) for the full sequence.

### colab_fine_tune

Parameters, all required:

| Parameter | Values | Description |
| --- | --- | --- |
| `MODEL` | `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-3B-Instruct` | Model to fine-tune. |
| `CONFIG` | path, for example `configs/train.yaml` | Training config: the training samples, and the LoRA and training settings. |

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

Lint, formatting, types, import layers and tests, the same steps CI runs on every push:

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

Start a session with a T4 GPU. The first call opens a browser to sign in:

```bash
uv run colab new -s qlora --gpu T4
```

The session is called `qlora`, as in every command below.

### 3. Evaluate the baseline models

Generate the predictions of each baseline model on the dev and test splits. The same notebook runs for each model and split:

```bash
uv run colab exec -s qlora -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct --env SPLIT=dev --env RUN=baseline --env BATCH_SIZE=16
uv run colab exec -s qlora -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct --env SPLIT=test --env RUN=baseline --env BATCH_SIZE=16
uv run colab exec -s qlora -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=Qwen/Qwen2.5-Coder-3B-Instruct --env SPLIT=dev --env RUN=baseline --env BATCH_SIZE=16
uv run colab exec -s qlora -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=Qwen/Qwen2.5-Coder-3B-Instruct --env SPLIT=test --env RUN=baseline --env BATCH_SIZE=16
```

The predictions and their metadata are in `/content/qwen2.5-coder-text2sql-qlora/predictions_colab/` on the session.

### 4. Download the predictions

Download the predictions and their metadata from the session. Files on the session are lost when it stops:

```bash
mkdir -p predictions
for name in qwen2.5-coder-1.5b-instruct_baseline_dev qwen2.5-coder-1.5b-instruct_baseline_test \
            qwen2.5-coder-3b-instruct_baseline_dev qwen2.5-coder-3b-instruct_baseline_test; do
  for ext in jsonl meta.json; do
    uv run colab download -s qlora /content/qwen2.5-coder-text2sql-qlora/predictions_colab/$name.$ext predictions/$name.$ext
  done
done
```

The files are in `predictions/`.

### 5. Score the baseline predictions

Score the baseline predictions. The test predictions need `--split test`, and their scores have no TS, as the test split has no test-suite databases:

```bash
uv run python -m text2sql.score predictions/qwen2.5-coder-1.5b-instruct_baseline_dev.jsonl
uv run python -m text2sql.score predictions/qwen2.5-coder-1.5b-instruct_baseline_test.jsonl --split test
uv run python -m text2sql.score predictions/qwen2.5-coder-3b-instruct_baseline_dev.jsonl
uv run python -m text2sql.score predictions/qwen2.5-coder-3b-instruct_baseline_test.jsonl --split test
```

The scores are in `scores/`.

### 6. Fine-tune the models

Fine-tune each model on the Spider training data. The same notebook runs for each model:

```bash
uv run colab exec -s qlora -f notebooks/colab_fine_tune.ipynb --timeout 28800 \
  --env MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct --env CONFIG=configs/train.yaml
uv run colab exec -s qlora -f notebooks/colab_fine_tune.ipynb --timeout 28800 \
  --env MODEL=Qwen/Qwen2.5-Coder-3B-Instruct --env CONFIG=configs/train.yaml
```

The adapters, their metadata and their training logs are in `/content/qwen2.5-coder-text2sql-qlora/adapters_colab/` on the session.

### 7. Download the adapters

Download the adapters from the session, each with its metadata and training log. Files on the session are lost when it stops:

```bash
for name in qwen2.5-coder-1.5b-instruct qwen2.5-coder-3b-instruct; do
  mkdir -p adapters/$name
  for file in adapter_model.safetensors adapter_config.json README.md tokenizer.json tokenizer_config.json chat_template.jinja training_args.bin training.meta.json training.log.json; do
    uv run colab download -s qlora /content/qwen2.5-coder-text2sql-qlora/adapters_colab/$name/$file adapters/$name/$file
  done
done
```

The files are in `adapters/`.

### 8. Upload the adapters to Hugging Face

Upload each adapter to its own Hugging Face repository, private to start with, so that the Colab session can read it. Log in once first, and replace `<user>` with your Hugging Face username:

```bash
uv run hf auth login
uv run hf upload <user>/qwen2.5-coder-1.5b-text2sql-qlora adapters/qwen2.5-coder-1.5b-instruct . \
  --private --exclude "training_args.bin"
uv run hf upload <user>/qwen2.5-coder-3b-text2sql-qlora adapters/qwen2.5-coder-3b-instruct . \
  --private --exclude "training_args.bin"
```

The adapters are at `huggingface.co/<user>/qwen2.5-coder-1.5b-text2sql-qlora` and `huggingface.co/<user>/qwen2.5-coder-3b-text2sql-qlora`. The published ones of this project are listed under [Adapters](#adapters).

### 9. Evaluate the fine-tuned models

Generate the predictions of each fine-tuned model on the dev and test splits. The same notebook as for the baseline runs for each model and split, with the adapter on top of the model and another run name:

```bash
uv run colab exec -s qlora -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct --env SPLIT=dev --env RUN=finetuned --env BATCH_SIZE=16 \
  --env ADAPTER=<user>/qwen2.5-coder-1.5b-text2sql-qlora
uv run colab exec -s qlora -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct --env SPLIT=test --env RUN=finetuned --env BATCH_SIZE=16 \
  --env ADAPTER=<user>/qwen2.5-coder-1.5b-text2sql-qlora
uv run colab exec -s qlora -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=Qwen/Qwen2.5-Coder-3B-Instruct --env SPLIT=dev --env RUN=finetuned --env BATCH_SIZE=16 \
  --env ADAPTER=<user>/qwen2.5-coder-3b-text2sql-qlora
uv run colab exec -s qlora -f notebooks/colab_generate_predictions.ipynb --timeout 3600 \
  --env MODEL=Qwen/Qwen2.5-Coder-3B-Instruct --env SPLIT=test --env RUN=finetuned --env BATCH_SIZE=16 \
  --env ADAPTER=<user>/qwen2.5-coder-3b-text2sql-qlora
```

The predictions and their metadata are in `/content/qwen2.5-coder-text2sql-qlora/predictions_colab/` on the session.

### 10. Download the predictions of the fine-tuned models

Download the predictions and their metadata from the session. Files on the session are lost when it stops:

```bash
for name in qwen2.5-coder-1.5b-instruct_finetuned_dev qwen2.5-coder-1.5b-instruct_finetuned_test \
            qwen2.5-coder-3b-instruct_finetuned_dev qwen2.5-coder-3b-instruct_finetuned_test; do
  for ext in jsonl meta.json; do
    uv run colab download -s qlora /content/qwen2.5-coder-text2sql-qlora/predictions_colab/$name.$ext predictions/$name.$ext
  done
done
```

The files are in `predictions/`, next to the baseline's.

### 11. Score the fine-tuned predictions

Score the fine-tuned predictions, with `--split test` for the test predictions:

```bash
uv run python -m text2sql.score predictions/qwen2.5-coder-1.5b-instruct_finetuned_dev.jsonl
uv run python -m text2sql.score predictions/qwen2.5-coder-1.5b-instruct_finetuned_test.jsonl --split test
uv run python -m text2sql.score predictions/qwen2.5-coder-3b-instruct_finetuned_dev.jsonl
uv run python -m text2sql.score predictions/qwen2.5-coder-3b-instruct_finetuned_test.jsonl --split test
```

The scores are in `scores/`, next to the baseline's.

### 12. Close the session

Stop the session:

```bash
uv run colab stop -s qlora
```
