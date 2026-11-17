"""Chat messages and the slice of history replayed to the LLM."""

from __future__ import annotations

from digital_twin.conversation import ChatMessage, count_questions, recent_exchanges


def exchange(n: int, *, error: bool = False) -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content=f"question {n}"),
        ChatMessage(role="assistant", content=f"answer {n}", error=error),
    ]


def history(*exchanges: list[ChatMessage]) -> list[ChatMessage]:
    return [message for pair in exchanges for message in pair]


def contents(pairs: list[tuple[ChatMessage, ChatMessage]]) -> list[tuple[str, str]]:
    return [(question.content, answer.content) for question, answer in pairs]


def test_message_defaults_describe_a_normal_reply():
    message = ChatMessage(role="assistant", content="hello")
    assert message.error is False
    assert message.sources == ()


def test_count_questions_counts_only_visitor_messages():
    assert count_questions([]) == 0
    assert count_questions(history(exchange(1), exchange(2, error=True), exchange(3))) == 3


def test_recent_exchanges_of_an_empty_history():
    assert recent_exchanges([], limit=3) == []


def test_recent_exchanges_pairs_questions_with_their_answers():
    pairs = recent_exchanges(history(exchange(1), exchange(2)), limit=3)
    assert contents(pairs) == [("question 1", "answer 1"), ("question 2", "answer 2")]


def test_recent_exchanges_keeps_only_the_latest_ones_oldest_first():
    pairs = recent_exchanges(history(*(exchange(n) for n in range(1, 7))), limit=3)
    assert contents(pairs) == [
        ("question 4", "answer 4"),
        ("question 5", "answer 5"),
        ("question 6", "answer 6"),
    ]


def test_recent_exchanges_skips_failed_answers():
    pairs = recent_exchanges(
        history(exchange(1), exchange(2, error=True), exchange(3), exchange(4, error=True)),
        limit=3,
    )
    assert contents(pairs) == [("question 1", "answer 1"), ("question 3", "answer 3")]


def test_failed_answers_do_not_use_up_the_limit():
    pairs = recent_exchanges(
        history(exchange(1), exchange(2), exchange(3, error=True), exchange(4, error=True)),
        limit=2,
    )
    assert contents(pairs) == [("question 1", "answer 1"), ("question 2", "answer 2")]


def test_a_limit_of_zero_replays_nothing():
    assert recent_exchanges(history(exchange(1), exchange(2)), limit=0) == []
