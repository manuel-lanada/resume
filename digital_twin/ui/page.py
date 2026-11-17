"""The page: one top-to-bottom script run of the Streamlit app."""

from __future__ import annotations

import logging

import streamlit as st

from digital_twin import settings
from digital_twin.ui import chat, resources, state
from digital_twin.ui.sidebar import render_index_status, render_sidebar

logger = logging.getLogger(__name__)


def embedder_error() -> str:
    return (
        f"I couldn't load my embedding model `{settings.EMBEDDING_MODEL}`, so I can't read "
        "the resume yet. Please try again in a few minutes. If this keeps happening, check "
        "the `EMBEDDING_MODEL` setting and the server log."
    )


def page_title() -> str:
    name = settings.CANDIDATE_NAME
    return f"Chat with {name}'s Digital Twin" if name else "Chat with my Digital Twin"


def render_header(title: str) -> None:
    st.title(f"👋 {title}")
    st.caption(
        "Welcome! I'm an AI twin that answers from my resume. Ask me about my "
        "experience, skills, projects and background."
    )


def render_setup_help(*, has_key: bool, has_resume: bool) -> None:
    """Friendly checklist shown when something required is missing."""
    st.info("**Almost there!** The Digital Twin needs two things before it can chat:")
    st.markdown(
        f"{'✅' if has_resume else '❌'} **A resume.** Put a file named `resume.pdf` next to "
        "`app.py` and reload, or upload a PDF from the sidebar to try one right now.\n\n"
        f"{'✅' if has_key else '❌'} **A Gemini API key.** Create a free key at "
        "[aistudio.google.com/apikey](https://aistudio.google.com/apikey), then set it as the "
        "`GEMINI_API_KEY` environment variable (or add `GEMINI_API_KEY=your-key` to a `.env` "
        "file next to `app.py`) and restart the app."
    )


def main() -> None:
    title = page_title()
    st.set_page_config(page_title=title, page_icon="💬", layout="centered")
    state.init()

    controls = render_sidebar()
    render_header(title)

    # Prerequisites: without a key and a resume there is nothing to chat about.
    api_key = resources.get_api_key()
    pdf_bytes = resources.read_resume(controls.uploaded_pdf)
    if not api_key or pdf_bytes is None:
        render_setup_help(has_key=bool(api_key), has_resume=pdf_bytes is not None)
        st.stop()

    # Load the embedding model on its own first, so a failed download or a bad
    # EMBEDDING_MODEL is not reported to the visitor as a problem with the PDF.
    try:
        resources.load_embedder()
    except Exception:
        logger.exception("Could not load the embedding model %r", settings.EMBEDDING_MODEL)
        st.error(embedder_error())
        st.stop()

    # Build the vector index, or fetch it from the cache.
    try:
        index = resources.load_index(pdf_bytes)
    except Exception as exc:
        st.error(f"I couldn't read that PDF. {exc}")
        st.stop()

    state.start_over_if_resume_changed(pdf_bytes)
    render_index_status(
        uploaded=controls.uploaded_pdf is not None, chunk_count=len(index.chunks)
    )

    chat.render_history(show_sources=controls.show_sources)
    chat.render_starter_questions()
    question = chat.read_question()
    if question:
        chat.answer_question(question, index, api_key, show_sources=controls.show_sources)
