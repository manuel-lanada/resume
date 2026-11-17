"""The chat area: history, starter questions, the input box and answering."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import streamlit as st
from google.genai import errors as genai_errors

from digital_twin import settings
from digital_twin.conversation import ChatMessage, count_questions
from digital_twin.llm import build_contents, friendly_error, stream_answer
from digital_twin.retrieval import ResumeIndex, retrieve
from digital_twin.ui import resources, state

logger = logging.getLogger(__name__)

SOURCES_LABEL = "Resume excerpts used"
EMPTY_ANSWER_NOTICE = (
    "I wasn't able to put an answer together for that. Could you rephrase the question?"
)
UNEXPECTED_ERROR_NOTICE = "Something went wrong on my side. Please try again."
QUESTION_LIMIT_NOTICE = (
    "That's the question limit for one session. Use 'Clear chat' in the sidebar to start over."
)


def md_safe(text: str) -> str:
    """Escape dollar signs so Streamlit does not render '$5M ... $8M' as LaTeX."""
    return text.replace("$", "\\$")


def md_restore(text: str) -> str:
    """Undo `md_safe`."""
    return text.replace("\\$", "$")


# --------------------------------------------------------------------------- #
# Conversation so far
# --------------------------------------------------------------------------- #

def render_history(*, show_sources: bool) -> None:
    for message in state.messages():
        with st.chat_message(message.role):
            st.markdown(md_safe(message.content))
            if show_sources and message.sources:
                _render_sources(message.sources)


def _render_sources(excerpts: Sequence[str]) -> None:
    with st.expander(SOURCES_LABEL):
        for excerpt in excerpts:
            st.text(excerpt)


# --------------------------------------------------------------------------- #
# Asking
# --------------------------------------------------------------------------- #

def render_starter_questions() -> None:
    """Offer a few opening questions, only before the first message."""
    if state.messages() or state.has_pending_question():
        return
    st.write("**Not sure where to start? Try one of these:**")
    columns = st.columns(2)
    for i, question in enumerate(settings.STARTER_QUESTIONS):
        columns[i % 2].button(
            question,
            key=f"starter_{i}",
            width="stretch",
            on_click=state.queue_question,
            args=(question,),
        )


def read_question() -> str | None:
    """Render the input box and return the question to answer on this run:
    a clicked starter question, else whatever was typed, else None."""
    limit_reached = count_questions(state.messages()) >= settings.MAX_QUESTIONS_PER_SESSION
    typed = st.chat_input(
        "Ask about my experience, skills or projects...",
        max_chars=settings.MAX_QUESTION_CHARS,
        disabled=limit_reached,
    )
    if limit_reached:
        st.info(QUESTION_LIMIT_NOTICE)
        state.pop_pending_question()
        return None

    question = (state.pop_pending_question() or typed or "").strip()
    return question or None


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #

def answer_question(
    question: str, index: ResumeIndex, api_key: str, *, show_sources: bool
) -> None:
    """Show the question, stream the twin's reply and save both to the history."""
    with st.chat_message("user"):
        st.markdown(md_safe(question))

    with st.chat_message("assistant"):
        reply = _stream_reply(question, index, api_key)
        if show_sources and reply.sources and not reply.error:
            _render_sources(reply.sources)

    state.record_exchange(question, reply)


def _stream_reply(question: str, index: ResumeIndex, api_key: str) -> ChatMessage:
    """Retrieve context, ask Gemini and stream the answer into the open chat
    bubble. Failures are shown to the visitor and returned as an error reply
    instead of being raised."""
    retrieved: list[str] = []
    try:
        retrieved = retrieve(question, index, resources.load_embedder())
        contents = build_contents(state.messages(), question, retrieved, profile=index.profile)
        pieces = stream_answer(resources.get_client(api_key), contents)
        shown = st.write_stream(md_safe(piece) for piece in pieces)
        answer = md_restore(shown).strip() if isinstance(shown, str) else ""
        failed = not answer
        if failed:
            answer = EMPTY_ANSWER_NOTICE
            st.markdown(answer)
    except genai_errors.APIError as exc:
        logger.exception("Gemini request failed")  # full details go to the server log
        answer, failed = friendly_error(exc), True
        st.warning(answer)
    except Exception:
        logger.exception("Unexpected error while answering")
        answer, failed = UNEXPECTED_ERROR_NOTICE, True
        st.warning(answer)

    return ChatMessage(role="assistant", content=answer, error=failed, sources=tuple(retrieved))
