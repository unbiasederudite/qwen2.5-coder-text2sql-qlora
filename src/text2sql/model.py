"""Qwen model loading and generation."""

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


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[Message],
    max_new_tokens: int = 256,
) -> str:
    """Generates the assistant reply to a conversation.

    Args:
        model (PreTrainedModel): Model from `load_model`.
        tokenizer (PreTrainedTokenizerBase): Tokenizer from `load_model`.
        messages (list[Message]): Conversation to reply to.
        max_new_tokens (int): Maximum number of new tokens.

    Returns:
        str: Reply without the prompt.
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0
    )
    return str(
        tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    )
