"""Fixtures shared by the whole test suite."""

from __future__ import annotations

import os

# A dummy key, set before the app is imported. python-dotenv never overrides a
# variable that already exists, so the real key in .env cannot leak into tests
# that are not explicitly marked `live`.
_KEY_FROM_ENVIRONMENT = os.environ.get("GEMINI_API_KEY", "")
os.environ["GEMINI_API_KEY"] = "test-key"

import pytest  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from google import genai  # noqa: E402

from digital_twin import settings  # noqa: E402
from tests.support import FakeEmbedder, FakeGeminiClient, make_pdf  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    """A two-page, text-based resume for a fictional person."""
    return make_pdf()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_gemini(monkeypatch: pytest.MonkeyPatch) -> FakeGeminiClient:
    """Replace google.genai.Client with a fake that records every request."""
    fake = FakeGeminiClient()
    monkeypatch.setattr(genai, "Client", fake)
    return fake


@pytest.fixture
def real_api_key() -> str:
    """The developer's real Gemini key, for tests marked `live`."""
    from_dotenv = dotenv_values(settings.PROJECT_ROOT / ".env").get("GEMINI_API_KEY") or ""
    key = (_KEY_FROM_ENVIRONMENT or from_dotenv).strip()
    if not key:
        pytest.skip("GEMINI_API_KEY is not set in the environment or in .env")
    return key
