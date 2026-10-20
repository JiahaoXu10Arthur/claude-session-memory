import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec

KNN = 4096  # sqlite-vec 的 KNN 一次最多取 4096 条


class Store:
    def __init__(self, path: str, dim: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)  # serve.py 多线程共用，core.Memory.lock 串行
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS obs (
                id INTEGER PRIMARY KEY, session_id TEXT, project TEXT, type TEXT,
                title TEXT, subtitle TEXT, narrative TEXT,
                facts TEXT, files TEXT, created_at TEXT, turn_id INTEGER)
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY, session_id TEXT, start INTEGER, project TEXT,
                created_at TEXT, title TEXT, text TEXT, UNIQUE(session_id, start))
        """)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cursors (path TEXT PRIMARY KEY, offset INTEGER)")
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS obs_fts USING fts5(title, subtitle, narrative, facts)"
        )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS obs_vec USING vec0("
            f"embedding float[{dim}] distance_metric=cosine, +obs_id integer)"
        )

    def add(self, batch: list[tuple[dict, np.ndarray]]) -> None:
        """没给 id 的记录按 max(id)+1 依次补上（写回 dict）"""
        with self.conn:
            self._insert_all(batch)

    def _insert_all(self, batch: list[tuple[dict, np.ndarray]]) -> None:
        (last,) = self.conn.execute("SELECT COALESCE(MAX(id), 0) FROM obs").fetchone()
        for o, vecs in batch:
            if o.get("id") is None:
                last += 1
                o["id"] = last
            self._insert(o, vecs)

    def replace_file(self, prefix: str, batch: list[tuple[dict, np.ndarray]]) -> None:
        """一个事务：title 以 prefix 开头（按字面、分大小写）的记录从三张表删掉，再插 batch"""
        with self.conn:
            ids = [r[0] for r in self.conn.execute("SELECT id FROM obs WHERE substr(title, 1, ?) = ?", (len(prefix), prefix))]
            for i in ids:
                self.conn.execute("DELETE FROM obs_vec WHERE obs_id = ?", (i,))
                self.conn.execute("DELETE FROM obs_fts WHERE rowid = ?", (i,))
                self.conn.execute("DELETE FROM obs WHERE id = ?", (i,))
            self._insert_all(batch)

    def _insert(self, o: dict, vecs: np.ndarray) -> None:
        """三张表各插一条，不 commit，调的人开事务"""
        self.conn.execute(
            "INSERT INTO obs (id, session_id, project, type, title, subtitle, "
            "narrative, facts, files, created_at, turn_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (o["id"], o["session_id"], o["project"], o["type"], o["title"],
             o["subtitle"], o["narrative"],
             json.dumps(o["facts"], ensure_ascii=False),
             json.dumps(o["files"], ensure_ascii=False), o["created_at"], o.get("turn_id")),
        )
        self.conn.execute(
            "INSERT INTO obs_fts (rowid, title, subtitle, narrative, facts) "
            "VALUES (?, ?, ?, ?, ?)",
            (o["id"], o["title"], o["subtitle"], o["narrative"],
             "\n".join(o["facts"])),
        )
        for v in vecs:
            self.conn.execute(
                "INSERT INTO obs_vec (embedding, obs_id) VALUES (?, ?)",
                (v.astype(np.float32).tobytes(), o["id"]),
            )

    def write(self, turn: dict, observations: list[dict], vecs: list[np.ndarray],
              cursor: tuple[str, int] | None) -> list[dict]:
        """一个事务：turn 没有就建、正文追加；observation 从 max(id)+1 起编号、挂上 turn_id、写三张表；
        cursor 给了就写游标。embedding 在外面做完再进来。返回带 id 的 observation"""
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO turns (session_id, start, project, created_at, title, text) VALUES (?, ?, ?, ?, ?, '')",
                (turn["session_id"], turn["start"], turn["project"], turn["created_at"], turn["title"]),
            )
            self.conn.execute("UPDATE turns SET text = text || ? WHERE session_id = ? AND start = ?",
                              (turn["text"], turn["session_id"], turn["start"]))
            (turn_id,) = self.conn.execute("SELECT id FROM turns WHERE session_id = ? AND start = ?",
                                           (turn["session_id"], turn["start"])).fetchone()
            (last,) = self.conn.execute("SELECT MAX(id) FROM obs").fetchone()
            for i, (o, v) in enumerate(zip(observations, vecs), 1):
                o["id"], o["turn_id"] = (last or 0) + i, turn_id
                self._insert(o, v)
            if cursor:
                self.conn.execute("INSERT OR REPLACE INTO cursors (path, offset) VALUES (?, ?)", cursor)
        return observations

    def search_dense(self, vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        rows = self.conn.execute(
            "SELECT obs_id, distance FROM obs_vec WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (vec.astype(np.float32).tobytes(), KNN),
        ).fetchall()
        seen = set()
        hits = []
        for obs_id, distance in rows:
            if obs_id in seen:
                continue
            seen.add(obs_id)
            hits.append((obs_id, distance))
            if len(hits) == k:
                break
        if len(hits) < k and len(rows) == KNN:
            raise RuntimeError(f"最近的 {KNN} 条向量只来自 {len(hits)} 条不同的记录，凑不满 {k} 条")
        return hits

    def search_bm25(self, query: str, k: int) -> list[tuple[int, float]]:
        fragments = [f for f in re.split(r"[^\w\x21-\x7e]+", query)
                     if any(c.isalnum() for c in f)]
        if not fragments:
            return []
        match = " OR ".join('"' + f.replace('"', '""') + '"' for f in fragments)
        return self.conn.execute(
            "SELECT rowid, bm25(obs_fts) FROM obs_fts WHERE obs_fts MATCH ? "
            "ORDER BY bm25(obs_fts), rowid LIMIT ?",
            (match, k),
        ).fetchall()

    def get(self, ids: list[int]) -> list[dict]:
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            "SELECT id, session_id, project, type, title, subtitle, narrative, "
            f"facts, files, created_at FROM obs WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {row[0]: row for row in rows}
        return [
            {
                "id": row[0], "session_id": row[1], "project": row[2], "type": row[3],
                "title": row[4], "subtitle": row[5], "narrative": row[6],
                "facts": json.loads(row[7]), "files": json.loads(row[8]),
                "created_at": row[9],
            }
            for row in (by_id[i] for i in ids)
        ]

    def cursor(self, path: str) -> int | None:
        row = self.conn.execute("SELECT offset FROM cursors WHERE path = ?", (path,)).fetchone()
        return row[0] if row else None

    def set_cursor(self, path: str, offset: int) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO cursors (path, offset) VALUES (?, ?)", (path, offset))

    def cursors(self) -> dict[str, int]:
        return dict(self.conn.execute("SELECT path, offset FROM cursors").fetchall())

    def del_cursor(self, path: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM cursors WHERE path = ?", (path,))

    def copy_from(self, path: str, ids: set[int]) -> None:
        """从另一个库把 ids 这些记录连同向量原样拷过来，不重新 embed；那个库只读打开。
        拷出来的库是要常驻读写的，顺手设成 WAL（这个设置留在文件里）"""
        self.conn.execute("PRAGMA journal_mode=WAL")
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        src.enable_load_extension(True)
        sqlite_vec.load(src)
        src.enable_load_extension(False)
        cols = "id, session_id, project, type, title, subtitle, narrative, facts, files, created_at"
        with self.conn:
            self.conn.executemany(
                f"INSERT INTO obs ({cols}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (r for r in src.execute(f"SELECT {cols} FROM obs ORDER BY id") if r[0] in ids),
            )
            self.conn.executemany(
                "INSERT INTO obs_fts (rowid, title, subtitle, narrative, facts) VALUES (?, ?, ?, ?, ?)",
                (r for r in src.execute(
                    "SELECT rowid, title, subtitle, narrative, facts FROM obs_fts ORDER BY rowid")
                 if r[0] in ids),
            )
            self.conn.executemany(
                "INSERT INTO obs_vec (embedding, obs_id) VALUES (?, ?)",
                (r for r in src.execute("SELECT embedding, obs_id FROM obs_vec ORDER BY rowid")
                 if r[1] in ids),
            )
        src.close()

    def stats(self) -> tuple[int, int]:
        row = self.conn.execute("SELECT COUNT(*), MAX(id) FROM obs").fetchone()
        return (row[0], row[1])
