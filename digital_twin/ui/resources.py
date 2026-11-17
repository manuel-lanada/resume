"""The app's expensive or secret dependencies.

The models and the resume index are built once per server process with
`st.cache_resource` and then shared by every visitor.
"""

from __future__ import annotations

import os

import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer

from digital_twin import settings
from digital_twin.ingestion import parse_resume
from digital_twin.llm import create_client
from digital_twin.retrieval import ResumeIndex, build_index

API_KEY_NAME = "GEMINI_API_KEY"


def get_api_key() -> str | None:
    """Look for the key in the environment first, then in Streamlit secrets."""
    key = os.getenv(API_KEY_NAME, "").strip()
    if key:
        return key
    try:
        return str(st.secrets[API_KEY_NAME]).strip() or None
    except Exception:  # no secrets.toml, or the key is not in it
        return None


def read_resume(uploaded_pdf: bytes | None) -> bytes | None:
    """Return the PDF to chat about: the visitor's upload if there is one,
    otherwise resume.pdf, otherwise None."""
    if uploaded_pdf is not None:
        return uploaded_pdf
    if settings.RESUME_PATH.exists():
        return settings.RESUME_PATH.read_bytes()
    return None


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> genai.Client:
    """Return the Gemini client for this key, creating it on first use."""
    return create_client(api_key)


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedder() -> SentenceTransformer:
    """Load the sentence-transformers model once per server process."""
    return SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")


@st.cache_resource(show_spinner="Reading and indexing the resume...", max_entries=5)
def load_index(pdf_bytes: bytes) -> ResumeIndex:
    """Parse, chunk, embed and index a resume.

    Streamlit caches the result keyed on the PDF bytes, so the default resume
    is processed once when the app boots and every chat turn afterwards only
    runs a fast in-memory search.
    """
    return build_index(parse_resume(pdf_bytes), load_embedder())
