# training

Fine-tuning: the training dataset, the trainer and its checkpoints.

| Path | Description |
| --- | --- |
| `data.py` | The training dataset: Spider samples as prompt-completion rows that fit the token limit. |
| `checkpoints.py` | The checkpoints of a training run, and what they say about it: unfinished or finished. |
| `train.py` | QLoRA fine-tuning on Spider. |
| `progress.py` | The progress bar of the training. |
