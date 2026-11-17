"""Embedding, indexing and nearest-chunk search (with a stand-in model)."""

from __future__ import annotations

import numpy as np
import pytest

from digital_twin import settings
from digital_twin.retrieval import ResumeIndex, build_index, embed, retrieve

CHUNKS = [
    "Jordan Avery, Senior Data Engineer based in Lisbon",
    "Programming languages: Python, SQL, Scala and Bash",
    "BSc in Computer Science from the University of Porto",
    "Certifications: AWS Certified Data Analytics",
    "Spoken languages: English, Portuguese and Spanish",
    "Hobbies: trail running and chess",
]


@pytest.fixture
def index(fake_embedder) -> ResumeIndex:
    return build_index(CHUNKS, fake_embedder)


def test_embed_returns_one_float32_row_per_text(fake_embedder):
    vectors = embed(fake_embedder, ["first text", "second text", "third text"])
    assert vectors.dtype == np.float32
    assert vectors.shape == (3, fake_embedder.dimension)


def test_embed_asks_the_model_for_normalised_numpy_vectors(fake_embedder):
    embed(fake_embedder, ["some text"])
    assert fake_embedder.options == {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }


def test_build_index_stores_every_chunk_and_vector(index, fake_embedder):
    assert index.chunks == tuple(CHUNKS)
    assert index.vectors.ntotal == len(CHUNKS)
    assert index.vectors.d == fake_embedder.dimension
    assert fake_embedder.calls == [CHUNKS], "all chunks should be embedded in one batch"


def test_profile_is_the_opening_chunk(index):
    assert index.profile == CHUNKS[0]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which programming languages do you know, Python or Scala?", CHUNKS[1]),
        ("Which University did you study Computer Science at?", CHUNKS[2]),
        ("Do you have any AWS certifications?", CHUNKS[3]),
        ("Do you enjoy chess or trail running?", CHUNKS[5]),
    ],
)
def test_retrieve_ranks_the_most_relevant_chunk_first(index, fake_embedder, question, expected):
    assert retrieve(question, index, fake_embedder)[0] == expected


def test_retrieve_returns_top_k_distinct_chunks_by_default(index, fake_embedder):
    results = retrieve("Tell me about yourself", index, fake_embedder)
    assert len(results) == settings.TOP_K
    assert len(set(results)) == settings.TOP_K
    assert set(results) <= set(CHUNKS)


def test_retrieve_honours_k(index, fake_embedder):
    assert len(retrieve("Python", index, fake_embedder, k=2)) == 2


def test_retrieve_from_an_index_smaller_than_k(fake_embedder):
    small = build_index(CHUNKS[:2], fake_embedder)
    results = retrieve("Python", small, fake_embedder, k=4)
    assert sorted(results) == sorted(CHUNKS[:2])


def test_retrieve_embeds_only_the_question(index, fake_embedder):
    retrieve("What do you do?", index, fake_embedder)
    assert fake_embedder.calls[-1] == ["What do you do?"]
