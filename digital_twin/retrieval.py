"""Semantic search over the resume: embeddings plus an in-memory FAISS index.

The embedding model is passed in rather than loaded here, so this module stays
cheap to import and can be exercised in tests with a stand-in model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import faiss
import numpy as np

from digital_twin import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class ResumeIndex:
    """A resume's chunks and the vector index over them.

    Row `i` of `vectors` is the embedding of `chunks[i]`.
    """

    chunks: tuple[str, ...]
    vectors: faiss.IndexFlatL2

    @property
    def profile(self) -> str:
        """The opening of the resume, which holds the name and headline."""
        return self.chunks[0]


def embed(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    """Embed texts as a float32 matrix, the dtype FAISS expects."""
    vectors = model.encode(
        list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype="float32")


def build_index(chunks: Sequence[str], model: SentenceTransformer) -> ResumeIndex:
    """Embed the chunks and load them into a FAISS index."""
    vectors = embed(model, chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return ResumeIndex(chunks=tuple(chunks), vectors=index)


def retrieve(
    question: str, index: ResumeIndex, model: SentenceTransformer, k: int = settings.TOP_K
) -> list[str]:
    """Return the k resume chunks closest to the question, nearest first."""
    _, ids = index.vectors.search(embed(model, [question]), min(k, len(index.chunks)))
    return [index.chunks[i] for i in ids[0] if i != -1]
