"""The Streamlit app, driven end to end through its UI.

`app.py` is run with Streamlit's AppTest harness. Only the two third-party
services are replaced: the embedding model and the Gemini client. Everything
else, from PDF parsing to FAISS to session state, is the real code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from google.genai import errors as genai_errors
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest

from digital_twin import settings
from digital_twin.ui import chat, page, resources
from tests.support import make_pdf

APP_FILE = settings.PROJECT_ROOT / "app.py"

OTHER_RESUME = make_pdf(
    [
        "Sam Rivera\nProduct Designer | Madrid, Spain\n\nEXPERIENCE\n"
        "Lumen Studio - Lead Product Designer (2019 - Present)\n"
        "Redesigned the checkout flow, lifting conversion by 18%."
    ]
)


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def app_environment(monkeypatch, fake_embedder, fake_gemini):
    """Point the app at the fakes and give every test fresh Streamlit caches.

    The developer's own configuration is shut out too: optional settings from
    the environment or .env (such as CANDIDATE_NAME), and any real
    secrets.toml, which AppTest would otherwise read.
    """
    st.cache_resource.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(settings, "CANDIDATE_NAME", "")
    monkeypatch.setattr(st, "secrets", empty_secrets())
    monkeypatch.setattr(resources, "SentenceTransformer", lambda *args, **kwargs: fake_embedder)
    yield
    st.cache_resource.clear()


def empty_secrets() -> Secrets:
    """A secrets store with nothing in it, whatever secrets.toml files exist.

    AppTest swaps in its own store only when a test sets `at.secrets`; this is
    the same mechanism it uses to build that store.
    """
    secrets = Secrets()
    secrets._secrets = {}
    return secrets


@pytest.fixture(autouse=True)
def resume_file(tmp_path, monkeypatch, sample_pdf) -> Path:
    """A default resume.pdf, as if it sat next to app.py."""
    path = tmp_path / "resume.pdf"
    path.write_bytes(sample_pdf)
    monkeypatch.setattr(settings, "RESUME_PATH", path)
    return path


def open_app() -> AppTest:
    return checked(AppTest.from_file(str(APP_FILE), default_timeout=30).run())


def ask(at: AppTest, question: str) -> AppTest:
    return checked(at.chat_input[0].set_value(question).run())


def checked(at: AppTest) -> AppTest:
    assert not at.exception, [error.value for error in at.exception]
    return at


def bubbles(at: AppTest) -> list[tuple[str, str]]:
    """(speaker, text) of every chat bubble on the page, top to bottom."""
    return [
        (message.name, [*message.markdown, *message.warning][0].value)
        for message in at.chat_message
    ]


def starter_labels(at: AppTest) -> list[str]:
    return sorted(button.label for button in at.main.button)


def history(at: AppTest) -> list:
    return at.session_state["messages"]


def api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "message": "simulated"}})


# --------------------------------------------------------------------------- #
# Setup checklist
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("has_key", "has_resume"), [(False, False), (True, False), (False, True)]
)
def test_setup_checklist_is_shown_until_key_and_resume_exist(
    monkeypatch, resume_file, has_key, has_resume
):
    if not has_key:
        monkeypatch.setenv("GEMINI_API_KEY", "")
    if not has_resume:
        resume_file.unlink()

    at = open_app()

    assert at.info[0].value.startswith("**Almost there!**")
    checklist = at.markdown[0].value
    assert f"{'✅' if has_resume else '❌'} **A resume.**" in checklist
    assert f"{'✅' if has_key else '❌'} **A Gemini API key.**" in checklist
    assert len(at.chat_input) == 0, "the chat must not be available yet"


def test_api_key_can_come_from_streamlit_secrets(monkeypatch, fake_gemini):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    at = AppTest.from_file(str(APP_FILE), default_timeout=30)
    at.secrets["GEMINI_API_KEY"] = "key-from-secrets"

    ask(checked(at.run()), "Who are you?")

    assert fake_gemini.constructed_with[-1]["api_key"] == "key-from-secrets"


def test_uploading_a_resume_satisfies_the_checklist(resume_file):
    resume_file.unlink()
    at = open_app()

    checked(at.sidebar.file_uploader[0].upload("cv.pdf", OTHER_RESUME, "application/pdf").run())

    assert len(at.info) == 0
    assert len(at.chat_input) == 1


# --------------------------------------------------------------------------- #
# First load
# --------------------------------------------------------------------------- #

def test_first_load_shows_header_starters_and_index_status():
    at = open_app()

    assert at.title[0].value == "👋 Chat with my Digital Twin"
    assert at.caption[0].value.startswith("Welcome! I'm an AI twin")
    assert starter_labels(at) == sorted(settings.STARTER_QUESTIONS)
    assert at.chat_input[0].placeholder == "Ask about my experience, skills or projects..."
    assert not at.chat_input[0].disabled
    assert bubbles(at) == []

    status = at.sidebar.caption[0].value
    assert status.startswith("Using resume.pdf: ")
    assert status.endswith(f" chunks indexed. Model: {settings.GEMINI_MODEL}.")


def test_title_uses_the_candidate_name_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "CANDIDATE_NAME", "Maria")
    assert open_app().title[0].value == "👋 Chat with Maria's Digital Twin"


# --------------------------------------------------------------------------- #
# Asking and answering
# --------------------------------------------------------------------------- #

def test_clicking_a_starter_question_asks_it(fake_gemini):
    fake_gemini.reply("I'm Jordan, ", "a data engineer.")
    at = open_app()

    checked(at.button(key="starter_1").click().run())

    question = settings.STARTER_QUESTIONS[1]
    assert bubbles(at) == [("user", question), ("assistant", "I'm Jordan, a data engineer.")]
    assert starter_labels(at) == [], "starters disappear once the chat has begun"
    assert fake_gemini.last_turns()[-1][1].endswith(f"VISITOR QUESTION: {question}")


def test_typed_question_is_trimmed_answered_and_remembered(fake_gemini):
    fake_gemini.reply("At Northwind I led a migration.")
    at = ask(open_app(), "   What did you do at Northwind?   ")

    assert bubbles(at) == [
        ("user", "What did you do at Northwind?"),
        ("assistant", "At Northwind I led a migration."),
    ]
    question, answer = history(at)
    assert (question.role, question.content) == ("user", "What did you do at Northwind?")
    assert (answer.role, answer.content) == ("assistant", "At Northwind I led a migration.")
    assert answer.error is False
    assert len(answer.sources) == settings.TOP_K


def test_conversation_is_redrawn_on_the_next_run(fake_gemini):
    at = ask(open_app(), "First question?")
    before = bubbles(at)
    assert bubbles(checked(at.run())) == before


def test_blank_question_is_ignored(fake_gemini):
    at = ask(open_app(), "     ")
    assert bubbles(at) == []
    assert history(at) == []
    assert fake_gemini.requests == []


def test_request_carries_resume_context_starting_with_the_profile(fake_gemini):
    ask(open_app(), "Which certifications do you hold?")

    [(role, prompt)] = fake_gemini.last_turns()
    assert role == "user"
    assert prompt.startswith("RESUME CONTEXT (excerpts from my resume):\n---\n")
    assert "Jordan Avery\nSenior Data Engineer" in prompt, "the profile chunk is always included"
    assert "AWS Certified Data Analytics" in prompt, "the relevant chunk was retrieved"
    assert prompt.endswith("VISITOR QUESTION: Which certifications do you hold?")


def test_follow_up_question_replays_the_conversation(fake_gemini):
    fake_gemini.reply("I maintain pipecheck.")
    at = ask(open_app(), "Tell me about your open-source work")
    ask(at, "Tell me more")

    turns = fake_gemini.last_turns()
    assert turns[:2] == [
        ("user", "Tell me about your open-source work"),
        ("model", "I maintain pipecheck."),
    ]
    assert turns[2][1].endswith("VISITOR QUESTION: Tell me more")


def test_dollar_signs_are_escaped_on_screen_but_stored_and_sent_verbatim(fake_gemini):
    fake_gemini.reply("I cut spend from $5M ", "to $3.8M.")
    at = ask(open_app(), "Did you save more than $1M?")

    assert bubbles(at) == [
        ("user", "Did you save more than \\$1M?"),
        ("assistant", "I cut spend from \\$5M to \\$3.8M."),
    ]
    assert history(at)[1].content == "I cut spend from $5M to $3.8M."

    ask(at, "How?")
    assert bubbles(at)[1] == ("assistant", "I cut spend from \\$5M to \\$3.8M."), "redrawn history"
    assert fake_gemini.last_turns()[1] == ("model", "I cut spend from $5M to $3.8M.")


def test_markdown_escaping_round_trips():
    for text in ("", "plain", "$5M to $8M", "already \\$ escaped", "$$"):
        assert chat.md_restore(chat.md_safe(text)) == text
    assert chat.md_safe("$5") == "\\$5"


# --------------------------------------------------------------------------- #
# Resume excerpts
# --------------------------------------------------------------------------- #

def test_excerpts_are_hidden_by_default():
    at = ask(open_app(), "Who are you?")
    assert len(at.expander) == 0


def test_excerpts_are_shown_when_switched_on():
    at = open_app()
    checked(at.sidebar.toggle[0].set_value(True).run())
    ask(at, "Which programming languages do you know?")

    [expander] = at.expander
    assert expander.label == "Resume excerpts used"
    assert [text.value for text in expander.text] == list(history(at)[1].sources)

    checked(at.run())  # and again when the history is redrawn
    assert [text.value for text in at.expander[0].text] == list(history(at)[1].sources)


def test_excerpts_are_not_shown_under_a_failed_answer(fake_gemini):
    at = open_app()
    checked(at.sidebar.toggle[0].set_value(True).run())
    fake_gemini.fail(api_error(503))

    ask(at, "Who are you?")

    assert len(at.expander) == 0


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (api_error(429), "I'm getting a lot of questions right now. Please try again in a minute."),
        (
            api_error(401),
            "My Gemini API key was rejected. Please check the `GEMINI_API_KEY` setting.",
        ),
        (api_error(500), "Sorry, I couldn't reach my language model just now. Please try again."),
        (RuntimeError("boom"), "Something went wrong on my side. Please try again."),
    ],
)
def test_failures_become_a_friendly_warning(fake_gemini, failure, message):
    fake_gemini.fail(failure)
    at = ask(open_app(), "Who are you?")

    assert at.warning[0].value == message
    assert bubbles(at) == [("user", "Who are you?"), ("assistant", message)]
    assert history(at)[1].error is True
    # Raw exception text stays in the server log, never on the page.
    assert "boom" not in message
    assert "simulated" not in message


@pytest.mark.parametrize("pieces", [(), ("   ", "\n")])
def test_empty_answer_asks_the_visitor_to_rephrase(fake_gemini, pieces):
    fake_gemini.reply(*pieces)
    at = ask(open_app(), "Who are you?")

    assert at.chat_message[1].markdown[-1].value == chat.EMPTY_ANSWER_NOTICE
    assert history(at)[1].content == chat.EMPTY_ANSWER_NOTICE
    assert history(at)[1].error is True


def test_failed_exchanges_are_left_out_of_later_requests(fake_gemini):
    at = ask(open_app(), "First question?")
    fake_gemini.fail(api_error(429))
    ask(at, "Second question fails?")
    ask(at, "Third question?")

    replayed = [text for _, text in fake_gemini.last_turns()[:-1]]
    assert replayed == ["First question?", "".join(fake_gemini.default_reply)]


def test_chat_keeps_working_after_a_failure(fake_gemini):
    fake_gemini.fail(api_error(503))
    at = ask(open_app(), "Who are you?")
    fake_gemini.reply("I'm Jordan.")

    ask(at, "Who are you?")

    assert bubbles(at)[-1] == ("assistant", "I'm Jordan.")


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #

def test_clear_chat_empties_the_conversation_and_brings_starters_back():
    at = ask(open_app(), "Who are you?")
    assert len(history(at)) == 2

    checked(at.sidebar.button[0].click().run())

    assert history(at) == []
    assert bubbles(at) == []
    assert starter_labels(at) == sorted(settings.STARTER_QUESTIONS)


def test_uploaded_resume_replaces_the_default_and_restarts_the_chat(fake_gemini):
    at = ask(open_app(), "Who are you?")

    checked(at.sidebar.file_uploader[0].upload("cv.pdf", OTHER_RESUME, "application/pdf").run())

    assert at.sidebar.caption[0].value.startswith("Using uploaded PDF: 1 chunks indexed.")
    assert history(at) == [], "a new resume starts a new conversation"

    ask(at, "Who are you?")
    prompt = fake_gemini.last_turns()[-1][1]
    assert "Sam Rivera" in prompt
    assert "Jordan Avery" not in prompt


def test_removing_the_upload_goes_back_to_the_default_resume():
    at = open_app()
    checked(at.sidebar.file_uploader[0].upload("cv.pdf", OTHER_RESUME, "application/pdf").run())

    checked(at.sidebar.file_uploader[0].clear().run())

    assert at.sidebar.caption[0].value.startswith("Using resume.pdf: ")


@pytest.mark.parametrize(
    ("pdf", "reason"),
    [
        (make_pdf([""]), "No selectable text was found in this PDF."),
        (b"this is not a pdf at all", "Stream has ended unexpectedly"),
    ],
)
def test_unreadable_upload_is_reported_and_blocks_the_chat(pdf, reason):
    at = open_app()

    checked(at.sidebar.file_uploader[0].upload("bad.pdf", pdf, "application/pdf").run())

    assert at.error[0].value.startswith(f"I couldn't read that PDF. {reason}")
    assert len(at.chat_input) == 0


def test_embedding_model_failure_is_not_blamed_on_the_pdf(monkeypatch, fake_embedder):
    def unreachable_hub(*args, **kwargs):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    monkeypatch.setattr(resources, "SentenceTransformer", unreachable_hub)
    at = open_app()

    assert [error.value for error in at.error] == [page.embedder_error()]
    assert "PDF" not in at.error[0].value
    assert "huggingface.co" not in at.error[0].value, "raw details belong in the server log"
    assert len(at.chat_input) == 0

    # A failed load is not cached, so the app recovers once the model loads.
    monkeypatch.setattr(resources, "SentenceTransformer", lambda *args, **kwargs: fake_embedder)
    checked(at.run())
    assert len(at.error) == 0
    assert len(at.chat_input) == 1


# --------------------------------------------------------------------------- #
# Limits and caching
# --------------------------------------------------------------------------- #

def test_question_limit_disables_the_input(monkeypatch):
    monkeypatch.setattr(settings, "MAX_QUESTIONS_PER_SESSION", 2)
    at = ask(open_app(), "Question one?")
    assert not at.chat_input[0].disabled

    ask(at, "Question two?")
    checked(at.run())

    assert at.chat_input[0].disabled
    assert at.info[0].value == chat.QUESTION_LIMIT_NOTICE
    assert len(history(at)) == 4

    checked(at.sidebar.button[0].click().run())   # "Clear chat" lifts the limit
    assert not at.chat_input[0].disabled


def test_chat_input_limits_the_question_length():
    assert open_app().chat_input[0].proto.max_chars == settings.MAX_QUESTION_CHARS


def test_models_and_index_are_built_once_and_reused(fake_gemini, fake_embedder):
    at = ask(open_app(), "First question?")
    ask(at, "Second question?")
    checked(at.run())

    assert len(fake_gemini.constructed_with) == 1, "one Gemini client per API key"
    resume_batches = [texts for texts in fake_embedder.calls if len(texts) > 1]
    assert len(resume_batches) == 1, "the resume is embedded once, then served from the cache"
    assert fake_embedder.calls[1:] == [["First question?"], ["Second question?"]]
