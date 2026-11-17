"""PDF ingestion: extract the resume text and split it into overlapping chunks."""

from __future__ import annotations

import io
import re

from pypdf import PdfReader

from digital_twin import settings

# Control characters and private-use glyphs (how many PDFs encode bullet
# symbols); they add noise to the embeddings and the prompt.
_NOISE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\ue000-\uf8ff]")
_SPACE_RUN = re.compile(r"[ \t\xa0]+")          # spaces, tabs, non-breaking spaces
_SPACE_AROUND_NEWLINE = re.compile(r" ?\n ?")
_BLANK_LINES = re.compile(r"\n{2,}")


def clean_text(text: str) -> str:
    """Strip noise characters and tidy up whitespace."""
    text = _NOISE.sub("", text)
    text = _SPACE_RUN.sub(" ", text)                # collapse runs of spaces/tabs
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)    # trim spaces around newlines
    text = _BLANK_LINES.sub("\n", text)             # drop blank lines
    return text.strip()


def extract_text(pdf_bytes: bytes) -> str:
    """Return all text in the PDF, with whitespace tidied up."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))


def chunk_text(
    text: str, size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP
) -> list[str]:
    """Split text into chunks of about `size` characters that overlap by about
    `overlap` characters, so a bullet point cut at a boundary still appears
    whole in one of the two neighbouring chunks. Cuts are moved to the nearest
    whitespace so words are never split in half.

    `overlap` may be at most half of `size`; beyond that the next chunk is not
    guaranteed to start after the previous one.
    """
    if size <= 0 or not 0 <= overlap <= size // 2:
        raise ValueError(
            f"Invalid chunking settings (size={size}, overlap={overlap}): size must be "
            "positive and overlap must be between 0 and half of size."
        )

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Back up to the last whitespace, unless that would shrink the
            # chunk by more than half (e.g. one very long unbroken string).
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        # Step back by `overlap`, then forward to the next word boundary.
        start = end - overlap
        boundary = re.search(r"\s", text[start:end])
        if boundary:
            start += boundary.end()
    return chunks


def parse_resume(pdf_bytes: bytes) -> list[str]:
    """Turn a resume PDF into the chunks that get embedded and indexed."""
    chunks = chunk_text(extract_text(pdf_bytes))
    if not chunks:
        raise ValueError(
            "No selectable text was found in this PDF. It may be a scanned image; "
            "export the resume as a text-based PDF and try again."
        )
    return chunks
