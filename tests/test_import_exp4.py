import tomllib

import numpy as np
import pytest

import core
import embed
import import_exp4
from import_exp4 import duplicates, narratives, open_ro
from store import Store


def vec(*xs):
    a = np.array(xs, dtype=np.float32)
    return a / np.linalg.norm(a)


def deg(d):
    """单位圆上转 d 度的二维向量：两条向量的余弦相似度就是夹角的余弦"""
    return vec(np.cos(np.radians(d)), np.sin(np.radians(d)))


def obs(id, session="s", created="2026-09-01T10:00:00.000Z", narrative="n", facts=()):
    return {"id": id, "session_id": session, "project": "p", "type": "decision", "title": "",
            "subtitle": "", "narrative": narrative, "facts": list(facts), "files": [],
            "created_at": created}


def row(id, session="s", created="2026-09-01T10:00:00.000Z", angle=0):
    return (id, session, created, deg(angle))


def test_narratives_take_the_first_vector_and_skip_empty_narratives(tmp_path):
    path = str(tmp_path / "src.db")
    Store(path, 2).add([
        (obs(1, narrative="n", facts=["f"]), np.stack([deg(0), deg(90)])),
        (obs(2, narrative="", facts=["f"]), np.stack([deg(45)])),
        (obs(3, created="2026-09-02T00:00:00.000Z"), np.stack([deg(30)])),
    ])
    got = narratives(open_ro(path))
    assert [(r[0], r[1], r[2]) for r in got] == [
        (1, "s", "2026-09-01T10:00:00.000Z"), (3, "s", "2026-09-02T00:00:00.000Z")]
    np.testing.assert_allclose(got[0][3], deg(0))
    np.testing.assert_allclose(got[1][3], deg(30))


def test_duplicates_merge_only_within_the_same_session_and_day():
    rows = [
        row(1, angle=0),
        row(2, angle=10),  # cos 10° = 0.985，同 session 同日 → 删
        row(3, angle=50),  # cos 50° = 0.643 → 留
        row(4, session="other", angle=0),  # 别的 session → 留
        row(5, created="2026-09-02T00:00:00.000Z", angle=0),  # 第二天 → 留
    ]
    assert duplicates(rows) == {2}


def test_duplicates_keep_the_earliest_created_at_then_the_smallest_id():
    # id 的顺序和时间的顺序故意相反，平手的两条也按 id 大的在前排：留 id 最小的会得到 {7, 8}，平手取先出现的会得到 {5, 7}
    rows = [
        row(5, created="2026-09-01T10:00:00.000Z"),
        row(8, created="2026-09-01T09:00:00.000Z"),
        row(7, created="2026-09-01T09:00:00.000Z"),
    ]
    assert duplicates(rows) == {5, 8}


def test_duplicates_follow_chains():
    # 1~2、2~3 都过 0.9，1 和 3 之间 cos 40° = 0.766 没过：三条算一组，留 1
    rows = [row(1, angle=0), row(2, angle=20), row(3, angle=40)]
    assert duplicates(rows) == {2, 3}


def test_duplicates_is_strictly_above_the_threshold():
    assert duplicates([row(1, angle=0), row(2, angle=0)], threshold=1.0) == set()


class FakeEmbedder:
    dim = 2

    def __init__(self, model_name):
        pass


def config(tmp_path):
    path = tmp_path / "exp4.toml"
    path.write_text(f'db = "{tmp_path / "exp4.db"}"\nunit = "observation"\n\n[embed]\nmodel = "fake"\n\n'
                    '[search]\nmode = "hybrid"\n')
    return str(path)


def test_main_copies_everything_but_the_duplicates(tmp_path, monkeypatch, capsys):
    src = str(tmp_path / "src.db")
    Store(src, 2).add([
        (obs(1, narrative="a", facts=["f"]), np.stack([deg(0), deg(90)])),
        (obs(2, narrative="a again"), np.stack([deg(5)])),
        (obs(3, narrative=""), np.stack([deg(5)])),  # narrative 空的不参与去重
        (obs(4, narrative="b"), np.stack([deg(80)])),
    ])
    monkeypatch.setattr(import_exp4, "SRC_DB", src)
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    path = config(tmp_path)
    import_exp4.main(path)
    assert capsys.readouterr().out.splitlines() == ["删 1 条", "(3, 4)"]
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    conn = core.Memory(cfg).store.conn
    assert conn.execute("SELECT id FROM obs ORDER BY id").fetchall() == [(1,), (3,), (4,)]
    assert conn.execute("SELECT rowid FROM obs_fts ORDER BY rowid").fetchall() == [(1,), (3,), (4,)]
    assert conn.execute("SELECT obs_id FROM obs_vec ORDER BY rowid").fetchall() == [(1,), (1,), (3,), (4,)]


def test_main_refuses_an_existing_db(tmp_path):
    path = config(tmp_path)
    (tmp_path / "exp4.db").write_bytes(b"")
    with pytest.raises(SystemExit, match="已经存在"):
        import_exp4.main(path)
