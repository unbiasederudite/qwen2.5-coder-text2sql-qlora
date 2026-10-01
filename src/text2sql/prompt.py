"""Chat messages for training and inference."""

from typing import TypedDict

from text2sql.data import Sample


class Message(TypedDict):
    """One chat message."""

    role: str  # "system", "user" or "assistant"
    content: str  # message text


def build_prompt(sample: Sample) -> list[Message]:
    """Builds the prompt messages.

    Args:
        sample (Sample): Sample to build from.

    Returns:
        list[Message]: System message with the schema, user message with the question.
    """
    return [
        Message(role="system", content="\n\n".join(sample["schema"])),
        Message(role="user", content=sample["question"]),
    ]


def build_completion(sample: Sample) -> list[Message]:
    """Builds the completion message.

    Args:
        sample (Sample): Sample to build from.

    Returns:
        list[Message]: Assistant message with the gold SQL.
    """
    return [Message(role="assistant", content=sample["query"])]
