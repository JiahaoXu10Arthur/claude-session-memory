import numpy as np

import embed
from embed import Embedder

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def test_minilm_vectors_are_normalized_float32():
    e = Embedder(MODEL)
    assert e.dim == 384
    v = e.embed(["stint gap fill", "删 stint 之后怎么处理"])
    assert v.shape == (2, 384)
    assert v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_empty_input():
    assert Embedder(MODEL).embed([]).shape == (0, 384)


def test_split_text_windows_overlap_and_stop_at_the_end():
    e = Embedder(MODEL)
    assert e.split_text(" ".join("abcdefghijklmnopqrstu"), 8, 2) == [
        "a b c d e f g h", "g h i j k l m n", "m n o p q r s t", "s t u"]


def test_split_text_short_text_is_one_chunk():
    e = Embedder(MODEL)
    assert e.split_text(" ".join("abcdefgh"), 8, 2) == ["a b c d e f g h"]
    assert e.split_text("a b c", 8, 2) == ["a b c"]
    assert e.split_text("", 8, 2) == []


def test_reranker_scores_the_matching_text_higher():
    r = embed.Reranker("BAAI/bge-reranker-v2-m3")
    scores = r.score("how do I kill a stale server process by pid",
                     ["kill_stale_server checks the process name behind the PID before killing it",
                      "the timeline gap after deleting a stint is left-filled"])
    assert scores.dtype == np.float32 and scores.shape == (2,)
    assert scores[0] > scores[1]
