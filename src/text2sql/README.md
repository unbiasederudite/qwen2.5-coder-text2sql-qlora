# text2sql

Text-to-SQL pipeline on Spider: data, prompts, fine-tuning, generation, scoring and metadata.

| File | Description |
| --- | --- |
| `spider.py` | Spider loaders. |
| `data.py` | Question and schema samples. |
| `prompt.py` | Chat messages. |
| `model.py` | Model loading and batch generation. |
| `generate.py` | Prediction generation. |
| `train.py` | QLoRA fine-tuning. |
| `meta.py` | Metadata of runs and reports. |
| `score.py` | Scoring of predictions. |
| `spider_evaluation/` | Vendored official Spider evaluation. |
