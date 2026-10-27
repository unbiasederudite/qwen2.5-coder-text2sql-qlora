"""The models: names for files, and loading."""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from text2sql.config import ModelConfig


def model_slug(model_name: str) -> str:
    """Shortens a model name for file and directory names.

    Args:
        model_name (str): Hugging Face model name.

    Returns:
        str: Name without the organization, in lower case.
    """
    return model_name.split("/")[-1].lower()


def load_model(
    name: str, config: ModelConfig, adapter: str | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads a quantized model and its tokenizer, with a LoRA adapter on top if given.

    Args:
        name (str): Hugging Face model name.
        config (ModelConfig): Config from `load_model_config`.
        adapter (str | None): Adapter directory, or Hugging Face repository of an adapter.

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizerBase]: Model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=BitsAndBytesConfig(**config["quantization"]),
        dtype=config["dtype"],
    )
    if adapter is not None:
        model.load_adapter(adapter)
    return model, AutoTokenizer.from_pretrained(name)
