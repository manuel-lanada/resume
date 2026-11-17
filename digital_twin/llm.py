"""Talking to Gemini: build the request, stream the answer, explain failures."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from google import genai
from google.genai import types

from digital_twin import settings
from digital_twin.conversation import ChatMessage, recent_exchanges
from digital_twin.prompts import SYSTEM_PROMPT, build_question_prompt


def create_client(api_key: str) -> genai.Client:
    """Create a Gemini client for the given API key."""
    return genai.Client(
        api_key=api_key, http_options={"api_version": settings.GEMINI_API_VERSION}
    )


def _text_content(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def build_contents(
    history: Sequence[ChatMessage], question: str, retrieved: Sequence[str], profile: str
) -> list[types.Content]:
    """Assemble the conversation for Gemini: recent turns, then the new
    question with its resume context attached."""
    contents: list[types.Content] = []

    # Earlier exchanges, so follow-ups like "tell me more" make sense.
    for asked, answered in recent_exchanges(history, settings.HISTORY_EXCHANGES):
        contents.append(_text_content("user", asked.content))
        contents.append(_text_content("model", answered.content))

    # The opening of a resume holds the name and headline. It is always
    # included so the twin knows who "I" is, whatever the search returned.
    excerpts = retrieved if profile in retrieved else [profile, *retrieved]
    contents.append(_text_content("user", build_question_prompt(question, excerpts)))
    return contents


def stream_answer(client: genai.Client, contents: list[types.Content]) -> Iterator[str]:
    """Yield the answer piece by piece as Gemini generates it."""
    stream = client.models.generate_content_stream(
        model=settings.GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=settings.TEMPERATURE,
        ),
    )
    for piece in stream:
        if piece.text:
            yield piece.text


def friendly_error(exc: Exception) -> str:
    """Turn an API failure into a message a visitor can understand."""
    code = getattr(exc, "code", None)
    if code in (401, 403) or "API key" in str(exc):
        return "My Gemini API key was rejected. Please check the `GEMINI_API_KEY` setting."
    if code == 404:
        return (
            f"The Gemini model `{settings.GEMINI_MODEL}` is not available for this API key. "
            "Set the `GEMINI_MODEL` environment variable to a model your key can use."
        )
    if code == 429:
        return "I'm getting a lot of questions right now. Please try again in a minute."
    return "Sorry, I couldn't reach my language model just now. Please try again."
