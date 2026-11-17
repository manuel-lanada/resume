"""PDF text extraction, cleaning and chunking."""

from __future__ import annotations

from itertools import pairwise

import pytest
from pypdf.errors import PdfReadError

from digital_twin import settings
from digital_twin.ingestion import chunk_text, clean_text, extract_text, parse_resume
from tests.support import make_pdf


def numbered_words(count: int) -> str:
    """'w0 w1 w2 ...': every word is unique, so chunks can be traced back."""
    return " ".join(f"w{i}" for i in range(count))


# --------------------------------------------------------------------------- #
# clean_text
# --------------------------------------------------------------------------- #

def test_clean_text_removes_control_characters():
    assert clean_text("a\x00b\x07c\x7fd\x0ce") == "abcde"


def test_clean_text_removes_private_use_glyphs_used_as_bullets():
    assert clean_text("\uf0b7 Led a team\n\ue000 Shipped it\uf8ff") == "Led a team\nShipped it"


def test_clean_text_keeps_real_bullets_and_punctuation():
    assert clean_text("• Saved $5M (40%) — fast") == "• Saved $5M (40%) — fast"


def test_clean_text_collapses_spaces_tabs_and_non_breaking_spaces():
    assert clean_text("Senior \t Data\xa0\xa0Engineer") == "Senior Data Engineer"


def test_clean_text_trims_spaces_around_newlines():
    assert clean_text("first line   \n   second line") == "first line\nsecond line"


def test_clean_text_drops_blank_lines_and_outer_whitespace():
    assert clean_text("\n\n  title \n\n\n body\n \n end \n\n") == "title\nbody\nend"


def test_clean_text_turns_windows_line_endings_into_newlines():
    assert clean_text("one\r\ntwo\r\n") == "one\ntwo"


# --------------------------------------------------------------------------- #
# chunk_text
# --------------------------------------------------------------------------- #

def test_text_shorter_than_one_chunk_is_returned_whole():
    assert chunk_text("A short resume.") == ["A short resume."]


@pytest.mark.parametrize("text", ["", "   ", "\n\n \n"])
def test_blank_text_gives_no_chunks(text):
    assert chunk_text(text) == []


def test_default_size_and_overlap_come_from_settings():
    text = numbered_words(2000)
    assert chunk_text(text) == chunk_text(
        text, size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP
    )


def test_chunks_never_exceed_the_requested_size():
    chunks = chunk_text(numbered_words(2000), size=200, overlap=40)
    assert len(chunks) > 10
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunks_are_cut_at_word_boundaries():
    text = numbered_words(2000)
    words = set(text.split())
    for chunk in chunk_text(text, size=200, overlap=40):
        assert set(chunk.split()) <= words, "a word was split in half"


def test_neighbouring_chunks_overlap():
    chunks = chunk_text(numbered_words(2000), size=200, overlap=40)
    for current, following in pairwise(chunks):
        shared = set(current.split()) & set(following.split())
        assert shared, "neighbouring chunks share no words"
        # The shared words are the tail of one chunk and the head of the next.
        assert current.split()[-1] in shared
        assert following.split()[0] in shared


def test_no_overlap_when_overlap_is_zero():
    chunks = chunk_text(numbered_words(500), size=100, overlap=0)
    assert len(chunks) > 5
    for current, following in pairwise(chunks):
        assert not set(current.split()) & set(following.split())


def test_every_word_survives_in_order():
    text = numbered_words(2000)
    seen: list[str] = []
    for chunk in chunk_text(text, size=200, overlap=40):
        for word in chunk.split():
            if word not in seen[-60:]:   # skip words repeated by the overlap
                seen.append(word)
    assert seen == text.split()


def test_text_without_whitespace_is_cut_at_the_size_limit():
    chunks = chunk_text("x" * 1200, size=500, overlap=100)
    assert [len(chunk) for chunk in chunks] == [500, 500, 400]


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-5, 0), (500, -1), (500, 251), (500, 500), (500, 900)],
)
def test_invalid_size_or_overlap_is_rejected(size, overlap):
    with pytest.raises(ValueError, match="Invalid chunking settings"):
        chunk_text("some text", size=size, overlap=overlap)


def test_overlap_of_exactly_half_the_size_is_allowed():
    chunks = chunk_text(numbered_words(500), size=100, overlap=50)
    assert len(chunks) > 5


# --------------------------------------------------------------------------- #
# extract_text / parse_resume
# --------------------------------------------------------------------------- #

def test_extract_text_reads_every_page(sample_pdf):
    text = extract_text(sample_pdf)
    assert text.startswith("Jordan Avery\nSenior Data Engineer | Lisbon, Portugal")
    assert "Northwind Logistics - Senior Data Engineer (2021 - Present)" in text   # page 1
    assert text.endswith("Spanish (conversational)")                               # page 2


def test_extract_text_has_no_blank_lines_or_stray_spaces(sample_pdf):
    text = extract_text(sample_pdf)
    assert "\n\n" not in text
    assert "  " not in text
    assert all(line == line.strip() for line in text.split("\n"))


def test_extract_text_of_a_pdf_without_text_is_empty():
    assert extract_text(make_pdf([""])) == ""


def test_parse_resume_returns_chunks_starting_with_the_profile(sample_pdf):
    chunks = parse_resume(sample_pdf)
    assert len(chunks) > 1
    assert chunks[0].startswith("Jordan Avery")
    assert all(len(chunk) <= settings.CHUNK_SIZE for chunk in chunks)


def test_parse_resume_rejects_a_pdf_without_selectable_text():
    with pytest.raises(ValueError, match="No selectable text was found"):
        parse_resume(make_pdf([""]))


@pytest.mark.parametrize("data", [b"this is not a pdf at all", b""])
def test_parse_resume_rejects_files_that_are_not_pdfs(data):
    with pytest.raises(PdfReadError):
        parse_resume(data)
