"""Request assembly, streaming and error messages for Gemini."""

from __future__ import annotations

import pytest
from google.genai import errors as genai_errors

from digital_twin import settings
from digital_twin.conversation import ChatMessage
from digital_twin.llm import build_contents, create_client, friendly_error, stream_answer
from digital_twin.prompts import SYSTEM_PROMPT, build_question_prompt

PROFILE = "Jordan Avery, Senior Data Engineer"


def turns(contents) -> list[tuple[str, str]]:
    """Flatten Gemini contents into (role, text) pairs."""
    assert all(len(content.parts) == 1 for content in contents)
    return [(content.role, content.parts[0].text) for content in contents]


def exchange(n: int, *, error: bool = False) -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content=f"question {n}"),
        ChatMessage(role="assistant", content=f"answer {n}", error=error),
    ]


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def test_question_prompt_layout():
    assert build_question_prompt("What do you do?", ["first excerpt", "second excerpt"]) == (
        "RESUME CONTEXT (excerpts from my resume):\n"
        "---\n"
        "first excerpt\n"
        "---\n"
        "second excerpt\n"
        "---\n"
        "\n"
        "VISITOR QUESTION: What do you do?"
    )


def test_system_prompt_states_the_ground_rules():
    assert SYSTEM_PROMPT.startswith('You are the AI "digital twin"')
    for rule in ("VOICE", "TONE", "GROUNDING", "MISSING INFORMATION", "SCOPE", "SAFETY", "HONESTY"):
        assert f". {rule}." in SYSTEM_PROMPT
    assert "\\" not in SYSTEM_PROMPT, "line continuations should not leak into the prompt"


# --------------------------------------------------------------------------- #
# build_contents
# --------------------------------------------------------------------------- #

def test_first_question_is_sent_as_a_single_user_turn():
    contents = build_contents([], "Who are you?", [PROFILE, "skills"], profile=PROFILE)
    assert turns(contents) == [("user", build_question_prompt("Who are you?", [PROFILE, "skills"]))]


def test_profile_is_put_first_when_the_search_missed_it():
    contents = build_contents([], "Which skills?", ["skills", "education"], profile=PROFILE)
    expected = build_question_prompt("Which skills?", [PROFILE, "skills", "education"])
    assert turns(contents)[-1] == ("user", expected)


def test_profile_is_not_repeated_when_the_search_found_it():
    contents = build_contents([], "Which skills?", ["skills", PROFILE], profile=PROFILE)
    expected = build_question_prompt("Which skills?", ["skills", PROFILE])
    assert turns(contents)[-1] == ("user", expected)


def test_earlier_exchanges_are_replayed_before_the_new_question():
    history = exchange(1) + exchange(2)
    contents = build_contents(history, "Tell me more", [PROFILE], profile=PROFILE)
    assert turns(contents) == [
        ("user", "question 1"),
        ("model", "answer 1"),
        ("user", "question 2"),
        ("model", "answer 2"),
        ("user", build_question_prompt("Tell me more", [PROFILE])),
    ]


def test_only_the_most_recent_exchanges_are_replayed():
    history = [message for n in range(1, 8) for message in exchange(n)]
    contents = build_contents(history, "And then?", [PROFILE], profile=PROFILE)
    replayed = [text for role, text in turns(contents)[:-1] if role == "user"]
    assert len(replayed) == settings.HISTORY_EXCHANGES
    assert replayed[-1] == "question 7"


def test_failed_exchanges_are_not_replayed():
    history = exchange(1) + exchange(2, error=True)
    contents = build_contents(history, "Try again", [PROFILE], profile=PROFILE)
    assert [text for _, text in turns(contents)[:-1]] == ["question 1", "answer 1"]


# --------------------------------------------------------------------------- #
# Client and streaming
# --------------------------------------------------------------------------- #

def test_create_client_passes_the_key_and_api_version(fake_gemini):
    client = create_client("my-key")
    assert client is fake_gemini
    assert fake_gemini.constructed_with == [
        {"api_key": "my-key", "http_options": {"api_version": settings.GEMINI_API_VERSION}}
    ]


def test_stream_answer_yields_text_pieces_and_skips_empty_ones(fake_gemini):
    fake_gemini.reply("I led ", "", "a team of $4M ", "engineers.")
    contents = build_contents([], "What did you lead?", [PROFILE], profile=PROFILE)
    assert list(stream_answer(fake_gemini, contents)) == ["I led ", "a team of $4M ", "engineers."]


def test_stream_answer_sends_the_model_prompt_and_temperature(fake_gemini):
    contents = build_contents([], "Who are you?", [PROFILE], profile=PROFILE)
    list(stream_answer(fake_gemini, contents))

    request = fake_gemini.last_request
    assert request["model"] == settings.GEMINI_MODEL
    assert request["contents"] is contents
    assert request["config"].system_instruction == SYSTEM_PROMPT
    assert request["config"].temperature == settings.TEMPERATURE


def test_stream_answer_lets_api_errors_through(fake_gemini):
    fake_gemini.fail(genai_errors.APIError(429, {"error": {"message": "slow down"}}))
    with pytest.raises(genai_errors.APIError):
        list(stream_answer(fake_gemini, []))


# --------------------------------------------------------------------------- #
# friendly_error
# --------------------------------------------------------------------------- #

def api_error(code: int, message: str = "simulated") -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "message": message}})


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (api_error(401), "My Gemini API key was rejected"),
        (api_error(403), "My Gemini API key was rejected"),
        (api_error(400, "API key not valid. Please pass a valid API key."), "API key was rejected"),
        (api_error(404), "is not available for this API key"),
        (api_error(429), "I'm getting a lot of questions right now"),
        (api_error(500), "I couldn't reach my language model"),
        (api_error(503), "I couldn't reach my language model"),
        (TimeoutError("timed out"), "I couldn't reach my language model"),
    ],
)
def test_friendly_error_explains_the_failure(error, expected):
    assert expected in friendly_error(error)


def test_model_not_found_message_names_the_model(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-retired")
    assert "`gemini-retired`" in friendly_error(api_error(404))


def test_friendly_error_never_shows_raw_api_details():
    secret_detail = "projects/123456/internal-quota-bucket"
    assert secret_detail not in friendly_error(api_error(429, secret_detail))
