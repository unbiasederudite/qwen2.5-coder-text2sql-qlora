# text2sql

The project package.

| Path | Description |
| --- | --- |
| `spider/` | The Spider dataset: files, databases, samples, predictions and scoring. |
| `training/` | Fine-tuning: the training dataset, the trainer and its checkpoints. |
| `meta/` | Metadata that describes generation, training and scoring runs. |
| `generation.py` | Generation of replies, and of the predictions of a model on a Spider split. |
| `score.py` | The score command, run as `python -m text2sql.score`. |
| `config.py` | The model and training config files, and their checks. |
| `model.py` | The models: names for files, and loading. |
| `prompt.py` | Chat messages for training and inference. |
| `jsonio.py` | Writing JSON files. |
