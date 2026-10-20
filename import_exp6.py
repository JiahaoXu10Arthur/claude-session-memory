"""实验 6 灌库。ours：用写入半把 9 个 session 的 transcript 压进 exp6-ours.db；theirs：把 claude-mem 同 9 个 session 的
observation 拷进 exp6-theirs.db，按 prompt_number 对到 turn。两边都先垫 exp4-dedup.db 的背景语料。
    uv run python import_exp6.py ours [session 前缀]     可以断点续跑：压过的 turn 按游标跳过，压了一半的删掉重压
    uv run python import_exp6.py theirs"""
import glob
import json
import os
import sqlite3
import sys
import time
import tomllib
from pathlib import Path

import core
import transcript
from import_claude_mem import SRC
from serve import CUT

ROOT = os.path.expanduser("~/.claude/projects/" + str(Path(__file__).resolve().parent).replace("/", "-") + "/")
BACKGROUND = "data/exp4-dedup.db"
SESSIONS = [  # 9/24 CUT 之后 session-mem 的 session，两边都有；本 session 51bceeb6 不放
    "9252576a-d51e-43b9-aefb-e5ce5c1daa99", "cc0feb44-abb6-4ebb-a2b6-e9a486cf4726",
    "5af1860a-a5aa-47b0-a6dd-9494a87ac246", "2948fed3-e69e-4a16-b192-42d2572a8b83",
    "5a7f9ca3-a7da-4688-b14d-33af6aa21e92", "81acef29-45ed-4c0e-88c2-bf84fe3070cc",
    "1ef4aebd-c3df-4d7a-b013-d8887ecc2492", "f1d28823-a294-4c4d-b710-76fe8adfaa8a",
    "aeb3f3c6-bc34-46b5-be29-4b874303a98b",
]


def background(mem: core.Memory) -> None:
    if mem.store.stats()[0]:
        return
    src = sqlite3.connect(f"file:{BACKGROUND}?mode=ro", uri=True)
    ids = {r[0] for r in src.execute("SELECT id FROM obs")}
    src.close()
    mem.store.copy_from(BACKGROUND, ids)
    print(f"背景语料 {len(ids)} 条", flush=True)


def drop_partial(conn: sqlite3.Connection, session_id: str, start: int) -> None:
    """上次跑到一半死在这个 turn：把它已经写的 observation 和 turn 删掉，从第 1 段重来"""
    row = conn.execute("SELECT id FROM turns WHERE session_id = ? AND start = ?", (session_id, start)).fetchone()
    if row is None:
        return
    ids = [r[0] for r in conn.execute("SELECT id FROM obs WHERE turn_id = ?", (row[0],))]
    with conn:
        for i in ids:
            conn.execute("DELETE FROM obs WHERE id = ?", (i,))
            conn.execute("DELETE FROM obs_fts WHERE rowid = ?", (i,))
            conn.execute("DELETE FROM obs_vec WHERE obs_id = ?", (i,))
        conn.execute("DELETE FROM turns WHERE id = ?", (row[0],))
    print(f"  上次压了一半：删 {len(ids)} 条重来", flush=True)


def compress_file(mem: core.Memory, path: str) -> None:
    store = mem.store
    cursor = store.cursor(path) or 0
    nxt = transcript.next_subagent if "/subagents/" in path else transcript.next_turn
    while True:
        r = nxt(path, cursor)
        if isinstance(r, int):
            if r != cursor:
                store.set_cursor(path, r)
            return
        cursor = r.end
        if r.created_at <= CUT:
            store.set_cursor(path, r.end)
            continue
        if "/subagents/" not in path:  # subagent 的段归进父 turn（同一个键，正文追加），turn 已存在不等于压了一半
            drop_partial(store.conn, r.session_id, r.start)
        segments = transcript.windows(r)
        meta = {"session_id": r.session_id, "start": r.start, "project": r.project,
                "created_at": r.created_at, "prompt": r.prompt}
        for i, seg in enumerate(segments, 1):
            t0 = time.time()
            out = mem.write(seg, {**meta, "cursor": (path, r.end) if i == len(segments) else None})
            print(f"  {r.session_id[:8]} start={r.start} 段 {i}/{len(segments)} {len(seg)} 字符 → {len(out)} 条，{time.time() - t0:.0f} 秒", flush=True)


def ours(prefix: str | None) -> None:
    with open("configs/exp6-ours.toml", "rb") as f:
        mem = core.Memory(tomllib.load(f))
    background(mem)
    for sid in SESSIONS:
        if prefix and not sid.startswith(prefix):
            continue
        print(f"{sid[:8]}", flush=True)
        for path in [ROOT + sid + ".jsonl", *sorted(glob.glob(ROOT + sid + "/subagents/*.jsonl"))]:
            compress_file(mem, path)


def theirs() -> None:
    with open("configs/exp6-theirs.toml", "rb") as f:
        mem = core.Memory(tomllib.load(f))
    background(mem)
    cm = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)
    q = ",".join("?" * len(SESSIONS))
    prompts = {(r[0], r[1]): r[2] for r in cm.execute(
        f"SELECT content_session_id, prompt_number, created_at FROM user_prompts WHERE content_session_id IN ({q})", SESSIONS)}
    rows = cm.execute(
        "SELECT o.id, s.content_session_id, o.project, o.type, o.title, o.subtitle, o.narrative, o.facts, "
        "o.files_read, o.files_modified, o.created_at, o.prompt_number FROM observations o "
        f"JOIN sdk_sessions s ON s.memory_session_id = o.memory_session_id WHERE s.content_session_id IN ({q}) ORDER BY o.id",
        SESSIONS).fetchall()
    cm.close()
    # turn 表照 B 边的键 (session_id, start) 建：起点从 transcript 切出来，prompt_number 按时间落进最近的前一个 turn
    conn = mem.store.conn
    turns: dict[str, list[tuple[str, int]]] = {}
    for sid in SESSIONS:
        path, cursor = ROOT + sid + ".jsonl", 0
        while True:
            r = transcript.next_turn(path, cursor)
            if isinstance(r, int):
                break
            cursor = r.end
            if r.created_at <= CUT:
                continue
            with conn:
                conn.execute("INSERT OR IGNORE INTO turns (session_id, start, project, created_at, title, text) VALUES (?, ?, ?, ?, ?, '')",
                             (sid, r.start, r.project, r.created_at, r.prompt[:core.TITLE_CHARS]))
            turns.setdefault(sid, []).append((r.created_at, r.start))
    obs, unmapped = [], 0
    for r in rows:
        sid, ts = r[1], prompts.get((r[1], r[11]))
        cand = [start for created, start in turns.get(sid, []) if ts and created <= ts]
        turn_id = None
        if cand:
            (turn_id,) = conn.execute("SELECT id FROM turns WHERE session_id = ? AND start = ?", (sid, cand[-1])).fetchone()
        else:
            unmapped += 1
        obs.append({"id": r[0], "session_id": sid, "project": r[2] or "", "type": r[3] or "", "title": r[4] or "",
                    "subtitle": r[5] or "", "narrative": r[6] or "", "facts": json.loads(r[7]) if r[7] else [],
                    "files": list(dict.fromkeys((json.loads(r[8]) if r[8] else []) + (json.loads(r[9]) if r[9] else []))),
                    "created_at": r[10], "turn_id": turn_id})
    mem.add(obs)
    print(f"claude-mem {len(obs)} 条，对不上 turn 的 {unmapped} 条", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "ours":
        ours(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        theirs()
