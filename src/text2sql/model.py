"""Qwen model loading and batch generation."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from text2sql.prompt import Message

MODELS = ["Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct"]
GENERATION = {
    "max_new_tokens": 256,  # the longest gold query is 202 Qwen tokens
    "do_sample": False,  # greedy
    "repetition_penalty": 1.0,
    "padding_side": "left",  # the replies must follow the prompts, not the padding
}


def load_model(name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads a model in 4-bit and its tokenizer.

    Args:
        name (str): Hugging Face model name.

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizerBase]: Model and tokenizer.
    """
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=quantization, dtype=torch.float16
    )
    return model, AutoTokenizer.from_pretrained(name)


def generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    conversations: list[list[Message]],
) -> list[str]:
    """Generates the assistant replies to several conversations at once.

    Decodes with the settings in `GENERATION` and sets the tokenizer to left padding.

    Args:
        model (PreTrainedModel): Model from `load_model`.
        tokenizer (PreTrainedTokenizerBase): Tokenizer from `load_model`.
        conversations (list[list[Message]]): Conversations to reply to.

    Returns:
        list[str]: Replies without the prompts, in the order of the conversations.
    """
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in conversations
    ]
    settings = dict(GENERATION)
    tokenizer.padding_side = settings.pop("padding_side")
    inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(
        model.device
    )
    output = model.generate(**inputs, **settings)
    return list(
        tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    )
