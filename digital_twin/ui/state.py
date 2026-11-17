"""Typed access to `st.session_state`, so the keys live in one place."""

from __future__ import annotations

import hashlib

import streamlit as st

from digital_twin.conversation import ChatMessage

_MESSAGES = "messages"                  # list[ChatMessage], oldest first
_PENDING_QUESTION = "pending_question"  # starter question waiting to be asked
_RESUME_ID = "resume_id"                # hash of the resume this chat is about


def init() -> None:
    """Make sure a new session has an (empty) conversation."""
    st.session_state.setdefault(_MESSAGES, [])


def messages() -> list[ChatMessage]:
    """The conversation so far."""
    return st.session_state[_MESSAGES]


def clear_messages() -> None:
    st.session_state[_MESSAGES] = []


def record_exchange(question: str, reply: ChatMessage) -> None:
    """Append a question and the reply it received to the conversation."""
    messages().extend([ChatMessage(role="user", content=question), reply])


def queue_question(question: str) -> None:
    """Button callback: hand a starter question to the chat loop."""
    st.session_state[_PENDING_QUESTION] = question


def has_pending_question() -> bool:
    return _PENDING_QUESTION in st.session_state


def pop_pending_question() -> str | None:
    """Take the queued starter question, if there is one."""
    return st.session_state.pop(_PENDING_QUESTION, None)


def start_over_if_resume_changed(pdf_bytes: bytes) -> None:
    """Start a fresh conversation whenever the resume changes."""
    resume_id = hashlib.sha256(pdf_bytes).hexdigest()
    if st.session_state.get(_RESUME_ID) != resume_id:
        st.session_state[_RESUME_ID] = resume_id
        clear_messages()
