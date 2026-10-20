"""实验 4：去重。把 data/exp3-obs.db 拷成新库，同 session 同日 narrative 向量余弦相似度大于 0.9 的一组只留最早的一条"""
import sqlite3
import sys
import tomllib
from pathlib import Path

import numpy as np
import sqlite_vec

import core

SRC_DB = "data/exp3-obs.db"
THRESHOLD = 0.9


def open_ro(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def narratives(conn: sqlite3.Connection) -> list[tuple[int, str, str, np.ndarray]]:
    """narrative 不为空的每条 observation 一项：(id, session_id, created_at, narrative 的向量)。
    narrative 的向量是这条 observation 的第一条向量（core.chunks 先放 narrative，再放 facts）"""
    first: dict[int, int] = {}
    for rowid, obs_id in conn.execute("SELECT rowid, obs_id FROM obs_vec ORDER BY rowid"):
        first.setdefault(obs_id, rowid)
    obs = conn.execute(
        "SELECT id, session_id, created_at FROM obs WHERE narrative != '' ORDER BY id").fetchall()
    rowids = [first[o[0]] for o in obs]
    vecs = dict(conn.execute(
        f"SELECT rowid, embedding FROM obs_vec WHERE rowid IN ({','.join('?' * len(rowids))})", rowids))
    return [(o[0], o[1], o[2], np.frombuffer(vecs[first[o[0]]], dtype=np.float32)) for o in obs]


def duplicates(rows: list[tuple[int, str, str, np.ndarray]], threshold: float = THRESHOLD) -> set[int]:
    """要删的 id。同 session 同日（created_at 前 10 字）里，narrative 向量余弦相似度大于 threshold 的两条算一组，
    组按传递并起来；每组留 created_at 最早的一条（并列取 id 小的），其余删"""
    parent = {r[0]: r[0] for r in rows}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_day: dict[tuple[str, str], list[tuple[int, str, str, np.ndarray]]] = {}
    for r in rows:
        by_day.setdefault((r[1], r[2][:10]), []).append(r)
    for group in by_day.values():
        vecs = np.stack([r[3] for r in group])
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        sim = vecs @ vecs.T
        for i, j in zip(*np.where(np.triu(sim, 1) > threshold)):
            parent[find(group[i][0])] = find(group[j][0])
    kept: dict[int, tuple[str, int]] = {}
    for r in rows:
        root = find(r[0])
        if root not in kept or (r[2], r[0]) < kept[root]:
            kept[root] = (r[2], r[0])
    return {r[0] for r in rows} - {k[1] for k in kept.values()}


def main(config_path: str) -> None:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    if Path(cfg["db"]).exists():
        sys.exit(f"{cfg['db']} 已经存在；要重拷先删掉它")
    src = open_ro(SRC_DB)
    ids = {i for (i,) in src.execute("SELECT id FROM obs")}
    dropped = duplicates(narratives(src))
    src.close()
    mem = core.Memory(cfg)
    mem.store.copy_from(SRC_DB, ids - dropped)
    print(f"删 {len(dropped)} 条")
    print(mem.store.stats())


if __name__ == "__main__":
    main(sys.argv[1])
