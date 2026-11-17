"""Opt-in tests against the real embedding model and the real Gemini API.

    pytest -m integration    real sentence-transformers model, no API calls
    pytest -m live           the whole app, including one real Gemini request
"""

from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from digital_twin import settings
from digital_twin.ingestion import parse_resume
from digital_twin.retrieval import build_index, retrieve


@pytest.fixture
def clean_caches():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.mark.integration
def test_real_embedding_model_finds_the_relevant_part_of_the_resume(sample_pdf):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
    index = build_index(parse_resume(sample_pdf), model)

    assert index.vectors.ntotal == len(index.chunks)
    for question, expected in [
        ("Which cloud certifications do you have?", "AWS Certified Data Analytics"),
        ("Where did you go to university?", "University of Porto"),
        ("How did you reduce cloud costs?", "$5M to $3.8M"),
    ]:
        retrieved = retrieve(question, index, model)
        assert len(retrieved) == settings.TOP_K
        assert any(expected in chunk for chunk in retrieved), question


@pytest.mark.live
def test_live_app_answers_from_the_resume(
    monkeypatch, tmp_path, sample_pdf, real_api_key, clean_caches
):
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(sample_pdf)
    monkeypatch.setattr(settings, "RESUME_PATH", resume)
    monkeypatch.setenv("GEMINI_API_KEY", real_api_key)

    at = AppTest.from_file(str(settings.PROJECT_ROOT / "app.py"), default_timeout=180).run()
    at.chat_input[0].set_value("What is your full name and your current job title?").run()

    assert not at.exception, [error.value for error in at.exception]
    answer = at.session_state["messages"][-1]
    print(f"\nGemini ({settings.GEMINI_MODEL}) answered: {answer.content}")
    assert not answer.error, answer.content
    assert "Jordan" in answer.content
    assert "Data Engineer" in answer.content
