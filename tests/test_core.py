import json
import threading
from pathlib import Path

import numpy as np
import pytest

import core
import embed

WORDS = ["stint", "gap", "race", "pid", "fold", "nfl"]


class FakeEmbedder:
    """按关键词在不在造向量；多一维常数，免得出现零向量。切块按 | 切，并记下每次调用的参数。"""

    def __init__(self, model_name):
        self.dim = len(WORDS) + 1
        self.split_calls = []

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, -1] = 0.1
            for j, w in enumerate(WORDS):
                if w in t.lower():
                    out[i, j] = 1.0
            out[i] /= np.linalg.norm(out[i])
        return out

    def split_text(self, text, size, overlap):
        self.split_calls.append((text, size, overlap))
        return text.split("|")


def obs(id, **kw):
    o = {"id": id, "session_id": "s", "project": "p", "type": "decision",
         "title": "", "subtitle": "", "narrative": "", "facts": [], "files": [],
         "created_at": "2026-09-24T00:00:00.000Z"}
    o.update(kw)
    return o


def make(tmp_path, monkeypatch, mode="dense"):
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    return core.Memory({"db": str(tmp_path / "m.db"),
                        "embed": {"model": "fake"},
                        "search": {"mode": mode}})


def test_chunks_are_narrative_then_each_fact():
    o = obs(1, title="T", subtitle="S", narrative="N", facts=["f1", "", "f2"])
    assert core.chunks(o, FakeEmbedder("fake")) == ["N", "f1", "f2"]


def test_chunks_skip_empty_narrative():
    assert core.chunks(obs(1, facts=["f1"]), FakeEmbedder("fake")) == ["f1"]


def test_chunks_of_a_turn_are_its_narrative_split_by_tokens():
    e = FakeEmbedder("fake")
    t = obs(1000001, type="turn", title="T", narrative="a|b", facts=["f"])
    assert core.chunks(t, e) == ["a", "b"]
    assert e.split_calls == [("a|b", 512, 64)]


def test_search_ranks_a_turn_by_its_best_chunk(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch)
    m.add([obs(1000001, type="turn", narrative="nfl race pid|stint gap"),
           obs(1000002, type="turn", narrative="stint")])
    assert [o["id"] for o in m.search("stint gap", 2)] == [1000001, 1000002]


def test_search_finds_observation_by_a_fact(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch)
    m.add([obs(1, narrative="race condition guard", facts=["pid check"]),
           obs(2, narrative="unrelated", facts=["stint gap fill left or right"])])
    assert [o["id"] for o in m.search("stint gap", 2)] == [2, 1]


def test_title_and_subtitle_are_not_embedded(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch)
    m.add([obs(1, title="stint gap", subtitle="stint gap", narrative="nfl"),
           obs(2, narrative="stint")])
    assert m.search("stint gap", 1)[0]["id"] == 2


def test_observation_without_text_is_stored_but_not_ranked(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch)
    m.add([obs(1, narrative="stint"), obs(2)])
    assert [o["id"] for o in m.search("stint", 5)] == [1]
    assert m.store.get([2])[0]["id"] == 2


def test_search_returns_full_observations(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch)
    o = obs(1, title="T", narrative="stint", facts=["gap"], files=["a.py"])
    m.add([o])
    assert m.search("stint", 1) == [o]


def test_unknown_search_mode_raises(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch, mode="nope")
    with pytest.raises(ValueError):
        m.search("stint", 1)
    with pytest.raises(ValueError):
        m.rankings("stint", 1)


def test_rrf_single_ranking_keeps_its_order():
    assert core.rrf([[3, 1, 2]], 60) == [3, 1, 2]


def test_rrf_adds_up_reciprocal_ranks():
    assert core.rrf([[1, 2], [2, 3]], 60) == [2, 1, 3]


def test_rrf_ties_go_to_the_earlier_ranking():
    assert core.rrf([[1], [2]], 60) == [1, 2]
    assert core.rrf([[2], [1]], 60) == [2, 1]


def test_rrf_rank_61_in_both_lists_beats_rank_1_in_one():
    assert core.rrf([[1, *range(100, 159), 2], [*range(200, 260), 2]], 60)[:2] == [2, 1]


def test_dense_mode_ranks_only_dense(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch)
    m.add([obs(1, narrative="stint")])
    assert list(m.rankings("stint", 5)) == ["dense"]


def test_hybrid_ranks_dense_then_bm25(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch, mode="hybrid")
    m.add([obs(1, narrative="stint gap"), obs(2, title="varlen")])
    assert list(m.rankings("varlen", 5).items()) == [("dense", [1]), ("bm25", [2])]


def test_hybrid_search_adds_what_only_bm25_finds(tmp_path, monkeypatch):
    hybrid = make(tmp_path, monkeypatch, mode="hybrid")
    hybrid.add([obs(1, narrative="stint gap"), obs(2, title="varlen")])
    dense = make(tmp_path, monkeypatch)
    assert [o["id"] for o in dense.search("varlen", 5)] == [1]
    assert [o["id"] for o in hybrid.search("varlen", 5)] == [1, 2]


def test_hybrid_search_returns_at_most_k(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch, mode="hybrid")
    m.add([obs(1, narrative="stint gap"), obs(2, title="varlen")])
    assert [o["id"] for o in m.search("varlen", 1)] == [1]


def test_hybrid_without_bm25_hits_keeps_dense_order(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch, mode="hybrid")
    m.add([obs(1, narrative="stint"), obs(2, narrative="stint gap")])
    assert m.rankings("，。", 5)["bm25"] == []
    assert [o["id"] for o in m.search("，。", 5)] == [1, 2]


def test_core_has_no_backend_words():
    text = (Path(__file__).parent.parent / "core.py").read_text().lower()
    for word in ["claude", "sqlite", "bge", "sentence_transformers"]:
        assert word not in text


class FakeReranker:
    """分数是问题里的词在文本里出现了几个；记下每次收到的文本"""

    def __init__(self, model_name):
        self.calls = []

    def score(self, query, texts):
        self.calls.append(texts)
        return np.array([sum(w in t for w in query.split()) for t in texts], dtype=np.float32)


def make_reranking(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(embed, "Reranker", FakeReranker)
    return core.Memory({"db": str(tmp_path / "m.db"),
                        "embed": {"model": "fake"},
                        "search": {"mode": "dense", "rerank": "fake-reranker"}})


def test_rerank_text_is_narrative_then_facts_without_title():
    assert core.rerank_text(obs(1, title="T", narrative="n", facts=["a", "b"])) == "n\na\nb"
    assert core.rerank_text(obs(1, title="T", narrative="", facts=["a"])) == "a"


def test_search_reranks_the_top_20_and_leaves_the_rest(tmp_path, monkeypatch):
    m = make_reranking(tmp_path, monkeypatch)
    m.add([obs(i, narrative="stint", facts=["gap"] if i in (20, 23) else []) for i in range(1, 26)])
    monkeypatch.setattr(m, "rankings", lambda query, k, store=None: {"dense": list(range(1, 26))})  # 融合前的顺序固定成 1..25
    ids = [o["id"] for o in m.search("stint gap", 25)]
    assert ids[0] == 20  # 前 20 里只有它的 fact 含 gap，重排到第 1
    assert ids[1:20] == [i for i in range(1, 20)]  # 其余 19 条同分，保持原顺序
    assert ids[20:] == [21, 22, 23, 24, 25]  # 23 的 fact 也含 gap，但在前 20 以外，不动
    assert m.reranker.calls == [["stint\ngap" if i == 20 else "stint" for i in range(1, 21)]]  # 收到的是 narrative 加 facts


def test_search_rerank_keeps_the_order_of_equal_scores(tmp_path, monkeypatch):
    m = make_reranking(tmp_path, monkeypatch)
    m.add([obs(i, narrative="stint") for i in range(1, 4)])
    monkeypatch.setattr(m, "rankings", lambda query, k, store=None: {"dense": [3, 1, 2]})
    assert [o["id"] for o in m.search("stint", 3)] == [3, 1, 2]
    assert m.reranker.calls == [["stint", "stint", "stint"]]  # 三条都打了分，而且分一样


def test_search_without_rerank_config_calls_no_reranker(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch)
    assert m.reranker is None


def test_rerank_text_limit_cuts_the_tail():
    assert core.rerank_text(obs(1, narrative="abcdef", facts=["gh"]), 4) == "abcd"
    assert core.rerank_text(obs(1, narrative="abcdef", facts=["gh"])) == "abcdef\ngh"


def test_search_passes_rerank_chars_to_the_reranker(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(embed, "Reranker", FakeReranker)
    m = core.Memory({"db": str(tmp_path / "m.db"), "embed": {"model": "fake"},
                     "search": {"mode": "dense", "rerank": "fake-reranker", "rerank_chars": 5}})
    m.add([obs(1, narrative="stint gap"), obs(2, narrative="stint")])
    monkeypatch.setattr(m, "rankings", lambda query, k, store=None: {"dense": [1, 2]})
    m.search("stint", 2)
    assert m.reranker.calls == [["stint", "stint"]]  # 「stint gap」被截成 5 个字


def test_format_is_one_line_per_hit_with_title_and_subtitle():
    hits = [obs(1, type="decision", project="p", created_at="2026-09-01T10:00:00.000Z", title="T1", subtitle="One sentence.", narrative="n" * 300),
            obs(2, type="bugfix", project="q", created_at="2026-09-02T10:00:00.000Z", title="T2", subtitle="", narrative="short")]
    assert core.format(hits) == "\n".join([
        "memory：和这句话相关的 2 条",
        "- [decision · p · 2026-09-01] T1 — One sentence.",
        "- [bugfix · q · 2026-09-02] T2",  # subtitle 空的只印 title
    ])


class FakeLLM:
    """记下收到的 prompt 和调用时锁有没有被拿着，按顺序回预设的答案"""

    def __init__(self, cfg):
        self.calls = []
        self.answers = []
        self.locked = []

    def complete(self, prompt, schema):
        self.calls.append((prompt, schema))
        self.locked.append(self.lock.locked())
        return self.answers.pop(0)


def writer(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(core.llm, "LLM", FakeLLM)
    m = core.Memory({"db": str(tmp_path / "m.db"), "embed": {"model": "fake"}, "search": {"mode": "dense"},
                     "llm": {"base_url": "http://x", "model": "m"}})
    m.llm.lock = m.lock
    return m


META = {"session_id": "sess", "start": 100, "project": "p", "created_at": "2026-09-25T18:29:24.819Z",
        "prompt": "可" * 80, "cursor": ("/t/sess.jsonl", 500)}


def test_write_compresses_stores_and_returns_observations(tmp_path, monkeypatch):
    m = writer(tmp_path, monkeypatch)
    m.llm.answers = [json.dumps({"observations": [
        {"title": "T", "subtitle": "S", "narrative": "stint gap", "facts": ["pid one", "race two"], "files": ["a.py"], "type": "decision"}]})]
    out = m.write("transcript body", META)
    prompt, schema = m.llm.calls[0]
    assert prompt == core.PROMPT + "transcript body" + core.PROMPT_TAIL and schema is core.SCHEMA
    assert m.llm.locked == [False]  # 模型调用一两分钟，不能拿着锁
    assert out == [{"id": 1, "turn_id": 1, "session_id": "sess", "project": "p", "created_at": META["created_at"],
                    "type": "decision", "title": "T", "subtitle": "S", "narrative": "stint gap",
                    "facts": ["pid one", "race two"], "files": ["a.py"]}]
    assert m.store.get([1])[0]["title"] == "T"
    assert [i for i, _ in m.store.search_dense(FakeEmbedder("f").embed(["race"])[0], 1)] == [1]  # fact 也进了向量
    assert m.store.conn.execute("SELECT title, text FROM turns").fetchall() == [("可" * 60, "transcript body")]
    assert m.store.conn.execute("SELECT * FROM cursors").fetchall() == [("/t/sess.jsonl", 500)]


@pytest.mark.parametrize("raw", ['{"observations": [{"title": "half', "", "null", '{"observations": null}',
                                 '{"observations": [{"title": "T"}]}'])
def test_write_with_a_malformed_answer_stores_nothing_but_still_moves_the_cursor(tmp_path, monkeypatch, capsys, raw):
    m = writer(tmp_path, monkeypatch)
    m.llm.answers = [raw]
    assert m.write("body", META) == []
    assert "模型输出不成形" in capsys.readouterr().err
    assert m.store.stats() == (0, None)
    assert m.store.conn.execute("SELECT text FROM turns").fetchall() == [("body",)]
    assert m.store.conn.execute("SELECT * FROM cursors").fetchall() == [("/t/sess.jsonl", 500)]


def test_write_without_cursor_does_not_move_it(tmp_path, monkeypatch):
    m = writer(tmp_path, monkeypatch)
    m.llm.answers = ['{"observations": []}']
    m.write("part 1", dict(META, cursor=None))
    assert m.store.conn.execute("SELECT COUNT(*) FROM cursors").fetchone() == (0,)


def test_write_propagates_a_model_failure_and_writes_nothing(tmp_path, monkeypatch):
    m = writer(tmp_path, monkeypatch)
    m.llm.answers = []  # pop 出错，当模型连不上
    with pytest.raises(IndexError):
        m.write("body", META)
    assert m.store.conn.execute("SELECT COUNT(*) FROM turns").fetchone() == (0,)
    assert m.store.conn.execute("SELECT COUNT(*) FROM cursors").fetchone() == (0,)


def test_search_and_write_share_one_lock(tmp_path, monkeypatch):
    m = writer(tmp_path, monkeypatch)
    m.llm.answers = ['{"observations": []}']
    m.lock.acquire()
    w = threading.Thread(target=m.write, args=("body", META))
    s = threading.Thread(target=m.search, args=("stint", 1))
    w.start()
    s.start()
    w.join(0.3)
    assert w.is_alive() and s.is_alive()  # 写库那段和搜索都在等锁
    assert m.llm.calls  # 模型调用不等锁，已经发出去了
    m.lock.release()
    w.join(5)
    s.join(5)
    assert not w.is_alive() and not s.is_alive()
    assert m.store.conn.execute("SELECT COUNT(*) FROM turns").fetchone() == (1,)


def test_prompt_and_schema_cover_the_nine_types():
    for t in core.TYPES:
        assert f"- {t}:" in core.PROMPT
    assert "Write in English" in core.PROMPT and "in English" in core.PROMPT_TAIL  # 中文进库 FTS 搜不到
    assert core.SCHEMA["properties"]["observations"]["items"]["properties"]["type"]["enum"] == core.TYPES
    assert core.SCHEMA["properties"]["observations"]["items"]["required"] == ["title", "subtitle", "narrative", "facts", "files", "type"]


def test_format_docs_prints_title_and_squeezed_narrative_head():
    hits = [obs(1, type="doc", title=".claude/adr/0001.md § A", narrative="line one\n\n  line   two " + "x" * 500)]
    out = core.format_docs(hits)
    lines = out.split("\n")
    assert lines[0] == "docs：和这句话相关的 1 节"
    assert lines[1] == "- [.claude/adr/0001.md § A]"
    assert lines[2].startswith("  line one line two xxx") and len(lines[2]) == 2 + 400


def test_search_can_use_another_store_and_skip_rerank(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "Reranker", lambda name: (_ for _ in ()).throw(AssertionError("rerank called")))
    m = make(tmp_path, monkeypatch)
    m.add([obs(1, narrative="stint")])
    import store as st
    other = st.Store(str(tmp_path / "docs.db"), m.embedder.dim)
    m.index(other, "d.md § ", [obs(None, type="doc", title="d.md § gap", narrative="gap", session_id="")])
    assert [h["title"] for h in m.search("gap", 5, store=other, rerank=False)] == ["d.md § gap"]
    assert [h["id"] for h in m.search("stint", 5)] == [1]  # 默认还是自己的 store


def test_index_replaces_the_files_sections(tmp_path, monkeypatch):
    m = make(tmp_path, monkeypatch)
    import store as st
    other = st.Store(str(tmp_path / "docs.db"), m.embedder.dim)
    m.index(other, "d.md § ", [obs(None, type="doc", title="d.md § A", narrative="stint", session_id="")])
    m.index(other, "d.md § ", [obs(None, type="doc", title="d.md § A", narrative="gap", session_id="")])
    assert [h["narrative"] for h in m.search("gap", 5, store=other, rerank=False)] == ["gap"]
    assert other.stats()[0] == 1
