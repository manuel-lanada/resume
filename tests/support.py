"""Test doubles and fixtures-as-data shared by the test suite.

Nothing here touches the network or loads a real model, so tests that use
these helpers are fast and deterministic.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# A fictional resume
# --------------------------------------------------------------------------- #

SAMPLE_RESUME_PAGES = (
    """\
Jordan Avery
Senior Data Engineer | Lisbon, Portugal
jordan.avery@example.com | linkedin.com/in/jordan-avery-example

SUMMARY
Data engineer with 9 years of experience building batch and streaming
pipelines, analytics platforms and machine learning infrastructure for
logistics and fintech companies. Comfortable leading small teams and
working directly with product and finance stakeholders.

EXPERIENCE
Northwind Logistics - Senior Data Engineer (2021 - Present)
•Led the migration of 140 nightly batch jobs from on-premise Hadoop to
  Spark on Kubernetes, cutting the daily processing window from 9 hours
  to 2.5 hours.
•Designed a streaming ingestion layer with Kafka and Flink that handles
  30,000 shipment events per second at peak.
•Reduced annual cloud spend from $5M to $3.8M by right-sizing clusters
  and introducing spot instances for non-critical workloads.
•Mentored 4 junior engineers and introduced design reviews for every
  new pipeline.

Brightpay Financial - Data Engineer (2017 - 2021)
•Built the company's first data warehouse on Snowflake with dbt, serving
  60 analysts across risk, finance and marketing.
•Implemented data quality checks with Great Expectations that caught
  schema drift before it reached regulatory reports.
•Automated month-end reconciliation, saving the finance team 3 days of
  manual work every month.""",
    """\
Atlas Analytics - Junior Data Analyst (2015 - 2017)
•Wrote SQL and Python reports for retail clients covering pricing,
  inventory and promotion effectiveness.
•Maintained a library of reusable Airflow operators used by the whole
  consulting team.

SKILLS
Languages: Python, SQL, Scala, Bash
Data: Spark, Flink, Kafka, Airflow, dbt, Snowflake, PostgreSQL
Cloud: AWS (S3, EMR, EKS, Lambda), Terraform, Docker, Kubernetes
Practices: data modelling, CI/CD, observability, cost optimisation

PROJECTS
Open-source: maintainer of "pipecheck", a small Python library for
validating pipeline configuration files, with 1,200 GitHub stars.
Hackathon: built a route-delay predictor in 48 hours that won the
Northwind internal hackathon in 2023.

EDUCATION
BSc in Computer Science, University of Porto (2011 - 2015)

CERTIFICATIONS
AWS Certified Data Analytics - Specialty (2022)
Databricks Certified Data Engineer Professional (2023)

LANGUAGES
English (fluent), Portuguese (native), Spanish (conversational)""",
)


# --------------------------------------------------------------------------- #
# PDF builder
# --------------------------------------------------------------------------- #

def make_pdf(pages: Sequence[str] = SAMPLE_RESUME_PAGES) -> bytes:
    """Build a minimal text-based PDF with one page per string.

    Hand-assembling the file keeps the test suite free of a PDF-writing
    dependency. Text is limited to Windows-1252 characters. A page made of an
    empty string has no extractable text, like a scanned resume.
    """
    font_number = 3
    page_numbers = [4 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{number} 0 R" for number in page_numbers)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for page_number, text in zip(page_numbers, pages, strict=True):
        stream = _content_stream(text)
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {page_number + 1} 0 R "
                f"/Resources << /Font << /F1 {font_number} 0 R >> >> >>"
            ).encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n%b\nendstream" % (len(stream), stream))

    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%b\nendobj\n" % (number, body)

    xref_offset = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        pdf += b"%010d 00000 n \n" % offset
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    return bytes(pdf)


def _content_stream(text: str) -> bytes:
    """PDF drawing commands that print `text` one line at a time."""
    commands = [b"BT", b"/F1 10 Tf", b"13 TL", b"54 750 Td"]
    if text:
        for line in text.split("\n"):
            commands.append(b"(" + _escape(line) + b") Tj")
            commands.append(b"T*")
    commands.append(b"ET")
    return b"\n".join(commands)


def _escape(line: str) -> bytes:
    data = line.encode("cp1252")
    return data.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


# --------------------------------------------------------------------------- #
# Fake embedding model
# --------------------------------------------------------------------------- #

class FakeEmbedder:
    """Stands in for SentenceTransformer: a normalised bag-of-words vector.

    Texts that share words land close together, which is enough to test that
    retrieval returns the most relevant chunk, without loading PyTorch.
    """

    dimension = 256

    def __init__(self) -> None:
        self.calls: list[list[str]] = []     # the texts of every encode() call
        self.options: dict[str, Any] = {}    # keyword arguments of the latest call

    def encode(self, sentences: list[str], **options: Any) -> np.ndarray:
        self.calls.append(list(sentences))
        self.options = options
        matrix = np.zeros((len(sentences), self.dimension), dtype="float64")
        for row, sentence in enumerate(sentences):
            for word in re.findall(r"[a-z0-9]+", sentence.lower()):
                # zlib.crc32 is stable between runs, unlike the built-in hash().
                matrix[row, zlib.crc32(word.encode()) % self.dimension] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


# --------------------------------------------------------------------------- #
# Fake Gemini client
# --------------------------------------------------------------------------- #

class FakeGeminiClient:
    """Stands in for google.genai.Client and records every request.

    Queue the outcome of each upcoming request with `reply()` / `fail()`.
    When nothing is queued the client answers with `default_reply`.

    The instance is callable and returns itself, so it can replace the
    `genai.Client` class: `monkeypatch.setattr(genai, "Client", fake)`.
    """

    default_reply: Sequence[str] = ("I have 9 years of experience ", "as a data engineer.")

    def __init__(self) -> None:
        self.constructed_with: list[dict[str, Any]] = []   # kwargs of each Client(...) call
        self.requests: list[dict[str, Any]] = []
        self._outcomes: list[Sequence[str] | Exception] = []
        self.models = SimpleNamespace(generate_content_stream=self._generate_content_stream)

    def __call__(self, **client_kwargs: Any) -> FakeGeminiClient:
        self.constructed_with.append(client_kwargs)
        return self

    def reply(self, *pieces: str) -> FakeGeminiClient:
        """Queue an answer, streamed as the given pieces."""
        self._outcomes.append(pieces)
        return self

    def fail(self, error: Exception) -> FakeGeminiClient:
        """Queue a failure raised when the stream is consumed."""
        self._outcomes.append(error)
        return self

    def _generate_content_stream(
        self, *, model: str, contents: list[Any], config: Any
    ) -> Iterator[SimpleNamespace]:
        self.requests.append({"model": model, "contents": contents, "config": config})
        outcome = self._outcomes.pop(0) if self._outcomes else self.default_reply
        if isinstance(outcome, Exception):
            raise outcome
        for piece in outcome:
            yield SimpleNamespace(text=piece)

    # Convenience accessors for assertions ---------------------------------- #

    @property
    def last_request(self) -> dict[str, Any]:
        return self.requests[-1]

    def last_turns(self) -> list[tuple[str, str]]:
        """The (role, text) pairs sent in the most recent request."""
        return [(content.role, content.parts[0].text) for content in self.last_request["contents"]]
