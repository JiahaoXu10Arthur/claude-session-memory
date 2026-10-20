from pathlib import Path

import numpy as np
import pytest

from store import Store


def vec(*xs):
    a = np.array(xs, dtype=np.float32)
    return a / np.linalg.norm(a)


def obs(id, **kw):
    o = {"id": id, "session_id": "s", "project": "p", "type": "decision",
         "title": "", "subtitle": "", "narrative": "", "facts": [], "files": [],
         "created_at": "2026-09-24T00:00:00.000Z"}
    o.update(kw)
    return o


NO_VECS = np.zeros((0, 3), dtype=np.float32)


def test_roundtrip(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    o = obs(7, title="T", narrative="n", facts=["a", "b"], files=["x.py"])
    s.add([(o, np.stack([vec(1, 0, 0), vec(0, 1, 0)]))])
    assert s.get([7]) == [o]
    assert s.stats() == (1, 7)


def test_reopen_existing_db(tmp_path):
    path = str(tmp_path / "s.db")
    Store(path, 3).add([(obs(7), np.stack([vec(1, 0, 0)]))])
    assert Store(path, 3).stats() == (1, 7)


def test_get_keeps_requested_order(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(i), NO_VECS) for i in (1, 2, 3)])
    assert [o["id"] for o in s.get([3, 1])] == [3, 1]


def test_search_dense_returns_distinct_observations(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    near = np.stack([vec(1, 0.01 * i, 0) for i in range(10)])
    s.add([(obs(1), near),
           (obs(2), np.stack([vec(1, 0.5, 0)])),
           (obs(3), np.stack([vec(0, 1, 0)]))])
    hits = s.search_dense(vec(1, 0, 0), 3)
    assert [i for i, _ in hits] == [1, 2, 3]
    assert hits[0][1] <= hits[1][1] <= hits[2][1]


def test_search_dense_ranks_observation_by_its_best_vector(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(1), np.stack([vec(0, 1, 0), vec(1, 0.1, 0)])),
           (obs(2), np.stack([vec(1, 0.3, 0)]))])
    assert [i for i, _ in s.search_dense(vec(1, 0, 0), 2)] == [1, 2]


def test_observation_without_vectors(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(5), NO_VECS)])
    assert s.get([5])[0]["id"] == 5
    assert s.search_dense(vec(1, 0, 0), 5) == []


def test_search_dense_raises_when_the_window_is_full_but_short_of_k(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(1), np.stack([vec(1, 0.001 * i, 0) for i in range(4096)])),
           (obs(2), np.stack([vec(0, 0, 1)]))])
    with pytest.raises(RuntimeError, match="凑不满 2 条"):
        s.search_dense(vec(1, 0, 0), 2)


def test_search_dense_full_window_is_fine_when_it_holds_k_records(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(1), np.stack([vec(1, 0.001 * i, 0) for i in range(4095)])),
           (obs(2), np.stack([vec(1, 0.5, 0)])),
           (obs(3), np.stack([vec(0, 0, 1)]))])
    assert [i for i, _ in s.search_dense(vec(1, 0, 0), 2)] == [1, 2]
    with pytest.raises(RuntimeError):
        s.search_dense(vec(1, 0, 0), 3)


def test_search_bm25_finds_keyword(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(1, narrative="kill_stale_server verifies the PID first"), NO_VECS),
           (obs(2, narrative="stint gap fill"), NO_VECS)])
    assert [i for i, _ in s.search_bm25("PID 对应的进程名", 5)] == [1]


def test_search_bm25_splits_on_chinese_punctuation(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(1, title='path bug produces false "从未存在过" claim'), NO_VECS),
           (obs(2, narrative="flash attn varlen kernel"), NO_VECS)])
    assert [i for i, _ in s.search_bm25("引号路径导致「从未存在过」误判那个", 5)] == [1]
    assert [i for i, _ in s.search_bm25("varlen、不留", 5)] == [2]


def test_search_bm25_keeps_ascii_compounds_as_phrases(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(1, narrative="kill_stale_server checks the PID"), NO_VECS),
           (obs(2, narrative="kill the server when the cache is stale"), NO_VECS)])
    assert [i for i, _ in s.search_bm25("kill_stale_server", 5)] == [1]


def test_search_bm25_keeps_hyphenated_terms_as_phrases(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(1, narrative="uses flash-attn varlen"), NO_VECS),
           (obs(2, narrative="flash memory and attn heads"), NO_VECS)])
    assert [i for i, _ in s.search_bm25("flash-attn", 5)] == [1]


def test_search_bm25_breaks_score_ties_by_id(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(2, narrative="stint gap"), NO_VECS),
           (obs(1, narrative="stint gap"), NO_VECS)])
    assert [i for i, _ in s.search_bm25("stint", 5)] == [1, 2]


def test_search_bm25_indexes_title_subtitle_and_facts(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(1, title="alpha"), NO_VECS),
           (obs(2, subtitle="beta"), NO_VECS),
           (obs(3, facts=["gamma"]), NO_VECS)])
    assert [i for i, _ in s.search_bm25("alpha", 5)] == [1]
    assert [i for i, _ in s.search_bm25("beta", 5)] == [2]
    assert [i for i, _ in s.search_bm25("gamma", 5)] == [3]


@pytest.mark.parametrize("q", ["flash-attn", 'say "hi"', "a (b", "col:x", "x*",
                               "NOT", "，。", "「从未存在过」", "", "varlen——不留", "a　b"])
def test_search_bm25_tolerates_special_characters(tmp_path, q):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(1, narrative="flash attn varlen"), NO_VECS)])
    s.search_bm25(q, 5)


def test_copy_from_copies_chosen_records_with_their_vectors(tmp_path):
    src_path = str(tmp_path / "src.db")
    one = obs(1, narrative="stint gap", facts=["left fill"])
    three = obs(3, title="varlen")
    src = Store(src_path, 3)
    src.add([(one, np.stack([vec(1, 0, 0), vec(0, 1, 0)])),
             (obs(2, narrative="race condition"), np.stack([vec(0, 0, 1)])),
             (three, np.stack([vec(1, 1, 0)]))])
    src.conn.close()
    before = Path(src_path).read_bytes()
    dst = Store(str(tmp_path / "dst.db"), 3)
    dst.copy_from(src_path, {1, 3})
    assert dst.stats() == (2, 3)
    assert dst.get([1, 3]) == [one, three]
    assert [i for i, _ in dst.search_dense(vec(0, 1, 0), 5)] == [1, 3]
    assert [i for i, _ in dst.search_bm25("varlen", 5)] == [3]
    assert dst.search_bm25("race", 5) == []
    assert Path(src_path).read_bytes() == before


def test_write_creates_turn_numbers_observations_and_moves_cursor(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    s.add([(obs(37233), NO_VECS)])
    turn = {"session_id": "sess", "start": 100, "project": "p", "created_at": "2026-09-25T18:29:24.819Z", "title": "可以，选 2", "text": "body one"}
    o1, o2 = obs(None, title="A"), obs(None, title="B")
    del o1["id"], o2["id"]
    out = s.write(turn, [o1, o2], [np.stack([vec(1, 0, 0)]), NO_VECS], ("/t/sess.jsonl", 500))
    assert [o["id"] for o in out] == [37234, 37235]
    assert s.get([37234])[0]["title"] == "A"
    assert s.conn.execute("SELECT turn_id FROM obs WHERE id IN (37234, 37235)").fetchall() == [(1,), (1,)]
    assert s.conn.execute("SELECT session_id, start, project, created_at, title, text FROM turns").fetchall() == [
        ("sess", 100, "p", "2026-09-25T18:29:24.819Z", "可以，选 2", "body one")]
    assert s.conn.execute("SELECT * FROM cursors").fetchall() == [("/t/sess.jsonl", 500)]
    assert [i for i, _ in s.search_dense(vec(1, 0, 0), 1)] == [37234]


def test_write_appends_to_an_existing_turn_and_replaces_the_cursor(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    turn = {"session_id": "sess", "start": 100, "project": "p", "created_at": "c", "title": "t", "text": "one\n"}
    s.write(turn, [], [], ("/t/sess.jsonl", 500))
    s.write(dict(turn, text="two"), [], [], ("/t/sess.jsonl", 900))  # 续篇：同一个键
    assert s.conn.execute("SELECT id, text FROM turns").fetchall() == [(1, "one\ntwo")]
    assert s.conn.execute("SELECT * FROM cursors").fetchall() == [("/t/sess.jsonl", 900)]
    assert s.stats() == (0, None)


def test_write_is_one_transaction(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    turn = {"session_id": "sess", "start": 100, "project": "p", "created_at": "c", "title": "t", "text": "x"}
    good, bad = obs(None), obs(None)
    del good["id"], bad["id"]
    bad["facts"] = object()  # 第 2 条 json.dumps 会炸，第 1 条已经插进去了
    with pytest.raises(TypeError):
        s.write(turn, [good, bad], [np.stack([vec(1, 0, 0)]), NO_VECS], ("/t/sess.jsonl", 500))
    assert s.stats() == (0, None)
    assert s.search_dense(vec(1, 0, 0), 1) == []
    assert s.conn.execute("SELECT COUNT(*) FROM turns").fetchone() == (0,)
    assert s.conn.execute("SELECT COUNT(*) FROM cursors").fetchone() == (0,)


def test_write_without_cursor_leaves_cursors_alone(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    turn = {"session_id": "sess", "start": 100, "project": "p", "created_at": "c", "title": "t", "text": "part 1"}
    s.write(turn, [], [], None)  # 不是这个 turn 的最后一段
    assert s.conn.execute("SELECT text FROM turns").fetchall() == [("part 1",)]
    assert s.conn.execute("SELECT COUNT(*) FROM cursors").fetchone() == (0,)


def test_copy_from_switches_the_new_db_to_wal_and_leaves_the_source_alone(tmp_path):
    src = str(tmp_path / "src.db")
    Store(src, 3).add([(obs(7), np.stack([vec(1, 0, 0)]))])
    before = Path(src).read_bytes()
    s = Store(str(tmp_path / "live.db"), 3)
    s.copy_from(src, {7})
    assert s.conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert s.stats() == (1, 7)
    assert Path(src).read_bytes() == before
    assert Store(src, 3).conn.execute("PRAGMA journal_mode").fetchone() == ("delete",)


def test_cursor_roundtrip(tmp_path):
    s = Store(str(tmp_path / "s.db"), 3)
    assert s.cursor("/t/a.jsonl") is None
    s.set_cursor("/t/a.jsonl", 40)
    s.set_cursor("/t/a.jsonl", 90)
    assert s.cursor("/t/a.jsonl") == 90


def doc(title, narrative="n"):
    return {"session_id": "", "project": "p", "type": "doc", "title": title, "subtitle": "",
            "narrative": narrative, "facts": [], "files": [title.split(" § ")[0]], "created_at": "2026-09-30T00:00:00Z"}


def dvec(n=1):
    return np.ones((n, 3), dtype=np.float32) / np.sqrt(3)


def test_add_assigns_ids_when_missing(tmp_path):
    s = Store(str(tmp_path / "d.db"), 3)
    a, b = doc("x.md § A"), doc("x.md § B")
    s.add([(a, dvec()), (b, dvec())])
    assert (a["id"], b["id"]) == (1, 2)
    c = doc("x.md § C")
    s.add([(c, dvec())])
    assert c["id"] == 3 and s.stats() == (3, 3)


def test_replace_file_swaps_one_files_sections_in_all_three_tables_and_escapes_like(tmp_path):
    s = Store(str(tmp_path / "d.db"), 3)
    s.add([(doc("a_b.md § A", "old one"), dvec()), (doc("a_b.md § B", "old two"), dvec(2)),
           (doc("axb.md § A", "keep me"), dvec()), (doc("a_b.md2 § A", "keep too"), dvec())])
    s.replace_file("a_b.md § ", [(doc("a_b.md § A", "new one"), dvec())])
    titles = sorted(r[0] for r in s.conn.execute("SELECT title FROM obs"))
    assert titles == ["a_b.md § A", "a_b.md2 § A", "axb.md § A"]
    assert s.conn.execute("SELECT narrative FROM obs WHERE title='a_b.md § A'").fetchone() == ("new one",)
    assert s.conn.execute("SELECT COUNT(*) FROM obs_vec").fetchone() == (3,)
    assert s.conn.execute("SELECT COUNT(*) FROM obs_fts").fetchone() == (3,)
    assert s.search_bm25("old", 10) == [] and len(s.search_bm25("new", 10)) == 1


def test_replace_file_with_an_empty_batch_just_deletes(tmp_path):
    s = Store(str(tmp_path / "d.db"), 3)
    s.add([(doc("a.md § A"), dvec()), (doc("b.md § A"), dvec())])
    s.replace_file("a.md § ", [])
    assert [r[0] for r in s.conn.execute("SELECT title FROM obs")] == ["b.md § A"]


def test_cursors_and_del_cursor(tmp_path):
    s = Store(str(tmp_path / "d.db"), 3)
    s.set_cursor("/r/a.md", 10)
    s.set_cursor("/r/b.md", 20)
    assert s.cursors() == {"/r/a.md": 10, "/r/b.md": 20}
    s.del_cursor("/r/a.md")
    assert s.cursors() == {"/r/b.md": 20}
