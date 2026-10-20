import sqlite3

from import_claude_mem import read_observations

COLS = ("id, memory_session_id, project, type, title, subtitle, narrative, "
        "facts, files_read, files_modified, created_at")


def make_src(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE observations ({COLS})")
    conn.executemany(f"INSERT INTO observations ({COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def row(id, **kw):
    r = {"id": id, "memory_session_id": "s1", "project": "p", "type": "bugfix",
         "title": "t", "subtitle": "st", "narrative": "n", "facts": '["f1"]',
         "files_read": "[]", "files_modified": "[]",
         "created_at": "2026-09-01T00:00:00.000Z"}
    r.update(kw)
    return tuple(r.values())


def test_reads_only_up_to_bound(tmp_path):
    src = str(tmp_path / "src.db")
    make_src(src, [row(10), row(20), row(30)])
    assert [o["id"] for o in read_observations(src, 20)] == [10, 20]


def test_maps_columns(tmp_path):
    src = str(tmp_path / "src.db")
    make_src(src, [row(1, title=None, narrative=None, facts='["f1", "f2"]',
                       files_read='["a.py", "b.py"]', files_modified='["b.py", "c.py"]')])
    assert read_observations(src, 1) == [{
        "id": 1, "session_id": "s1", "project": "p", "type": "bugfix",
        "title": "", "subtitle": "st", "narrative": "",
        "facts": ["f1", "f2"], "files": ["a.py", "b.py", "c.py"],
        "created_at": "2026-09-01T00:00:00.000Z"}]


def test_null_and_empty_json_columns(tmp_path):
    src = str(tmp_path / "src.db")
    make_src(src, [row(1, facts=None, files_read="", files_modified=None)])
    o = read_observations(src, 1)[0]
    assert o["facts"] == []
    assert o["files"] == []


def test_source_is_opened_read_only(tmp_path, monkeypatch):
    src = str(tmp_path / "src.db")
    make_src(src, [row(1)])
    seen = []
    real_connect = sqlite3.connect

    def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy)
    read_observations(src, 1)
    [(args, kwargs)] = seen
    assert "mode=ro" in args[0]
    assert kwargs.get("uri") is True
