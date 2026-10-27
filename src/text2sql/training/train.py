"""QLoRA fine-tuning on Spider."""

from pathlib import Path
from typing import TypedDict

from peft import LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from text2sql.config import TrainConfig
from text2sql.jsonio import write_json
from text2sql.spider.samples import Sample
from text2sql.training.checkpoints import has_checkpoint, remove_checkpoints
from text2sql.training.data import build_dataset


class SampleCounts(TypedDict):
    """How many samples a training run used."""

    samples: int  # samples trained on
    dropped_too_long: int  # samples dropped for being longer than `max_length`


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[Sample],
    adapter_dir: Path,
    config: TrainConfig,
) -> SampleCounts:
    """Fine-tunes the model with LoRA, resuming from a checkpoint if there is one.

    Args:
        model (PreTrainedModel): Model from `load_model`.
        tokenizer (PreTrainedTokenizerBase): Tokenizer from `load_model`.
        samples (list[Sample]): Samples to train on.
        adapter_dir (Path): Directory of the adapter.
        config (TrainConfig): Config from `load_train_config`.

    Returns:
        SampleCounts: Samples trained on and samples dropped for length.
    """
    dataset, dropped = build_dataset(samples, tokenizer, config["data"]["max_length"])
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(adapter_dir),
            max_length=config["data"]["max_length"],
            **config["training"],
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(**config["lora"]),
    )
    # TRL casts the adapter of a 4-bit model to bf16, which fp16 loss scaling cannot unscale
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    trainer.train(resume_from_checkpoint=has_checkpoint(adapter_dir))
    trainer.save_model(str(adapter_dir))
    write_json(adapter_dir / "training.log.json", trainer.state.log_history)
    remove_checkpoints(adapter_dir)
    return SampleCounts(samples=len(dataset), dropped_too_long=dropped)
