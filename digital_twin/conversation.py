"""The chat transcript: what a message is, and which messages the LLM sees."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One bubble in the chat.

    `error` and `sources` only ever apply to assistant messages: `error` marks
    a reply that is a failure notice rather than a real answer, and `sources`
    holds the resume excerpts that were retrieved for it.
    """

    role: Role
    content: str
    error: bool = False
    sources: tuple[str, ...] = ()


def count_questions(history: Sequence[ChatMessage]) -> int:
    """How many questions the visitor has asked so far."""
    return sum(1 for message in history if message.role == "user")


def recent_exchanges(
    history: Sequence[ChatMessage], limit: int
) -> list[tuple[ChatMessage, ChatMessage]]:
    """Return the last `limit` (question, answer) pairs worth replaying to the
    LLM, oldest first, so follow-ups like "tell me more" make sense.

    `history` alternates user, assistant, user, assistant. Exchanges that
    ended in an error are skipped.
    """
    if limit <= 0:
        return []
    pairs = [
        (question, answer)
        for question, answer in zip(history[0::2], history[1::2], strict=False)
        if not answer.error
    ]
    return pairs[-limit:]
