import glob
import json
import sqlite3
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import core
from import_claude_mem import SRC, UPTO

TRANSCRIPTS = "data/transcripts"  # 9/25 19:18 从 ~/.claude/projects/ 整个拷来
OBS_DB = "data/bge-m3.db"  # 实验 2 的库，A 边从这里拷
T0 = "2026-08-26T18:18:03.000Z"  # 拷 transcript 的时刻往前 30 天，这之后写的内容一条没删
CUT = "2026-09-24T15:56:30.162Z"  # id ≤ 37233 的最后一条 observation
TOOL_OUTPUT = 2000
FIRST_TURN_ID = 1000001
BATCH = 32


def norm(s: str) -> str:
    return " ".join(s.split())


def in_window(row: dict) -> bool:
    return T0 <= row.get("timestamp", "") <= CUT


def user_text(row: dict) -> str | None:
    """用户发的消息（含 task-notification）返回原文；tool 结果、meta、sidechain 和别的行返回 None"""
    if row.get("type") != "user" or row.get("isSidechain") or row.get("isMeta"):
        return None
    c = row["message"]["content"]
    if isinstance(c, str):
        return c
    if any(b.get("type") == "tool_result" for b in c):
        return None
    return "".join(b.get("text", "") for b in c if b.get("type") == "text")


def queued_prompt(row: dict) -> str | None:
    """排队发的 prompt 送进模型的那一行：attachment 里 type 为 queued_command"""
    a = row.get("attachment", {})
    if row.get("type") != "attachment" or a.get("type") != "queued_command":
        return None
    p = a.get("prompt", "")
    return p if isinstance(p, str) else "".join(b.get("text", "") for b in p if isinstance(b, dict))


def pieces(row: dict) -> list[str]:
    """一行 transcript 里放进 turn 的几段；attachment 只放排队送进去的 prompt"""
    if row.get("type") == "attachment":
        p = queued_prompt(row)
        return [] if p is None else [p]
    c = row.get("message", {}).get("content")
    out = []
    if row.get("type") == "user":
        if isinstance(c, str):
            return [c]
        for b in c or []:
            if b.get("type") == "tool_result":
                r = b.get("content")
                out.append((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False))[:TOOL_OUTPUT])
            elif b.get("type") == "text":
                out.append(b["text"])
    elif row.get("type") == "assistant":
        for b in c or []:
            if b.get("type") == "text":
                out.append(b["text"])
            elif b.get("type") == "thinking":
                out.append(b.get("thinking", ""))
            elif b.get("type") == "tool_use":
                out.append(b.get("name", "") + " " + json.dumps(b.get("input", {}), ensure_ascii=False))
    return out


def find_starts(rows: list[dict], prompts: list[tuple[int, str, str]]) -> list[tuple[int, int, str, str]]:
    """每句 prompt 的 turn 从 transcript 的哪一行开始，返回 (行号, prompt_number, created_at, prompt 原文)，按行号排。
    先找原话对得上、时间最近的用户消息（一天以内）；再找排队送进去的那一行；再找 120 秒内最近的用户消息。
    都找不到的 prompt 不开新 turn，它的内容留在前一个 turn 里"""
    users = [(i, datetime.fromisoformat(r["timestamp"]), norm(u)) for i, r in enumerate(rows)
             if "timestamp" in r and (u := user_text(r)) is not None]
    queued = [(i, norm(q)) for i, r in enumerate(rows) if (q := queued_prompt(r)) is not None]
    starts = []
    for n, text, at in prompts:
        key, t = norm(text)[:40], datetime.fromisoformat(at)
        same = [(abs((u - t).total_seconds()), i) for i, u, s in users if key and s.startswith(key)]
        if same and min(same)[0] <= 86400:
            starts.append((min(same)[1], n, at, text))
            continue
        q = [i for i, s in queued if key and s.startswith(key)]
        if q:
            starts.append((q[0], n, at, text))
            continue
        near = [(abs((u - t).total_seconds()), i) for i, u, _ in users]
        if near and min(near)[0] <= 120:
            starts.append((min(near)[1], n, at, text))
    return sorted(starts)


def session_turns(rows: list[dict], prompts: list[tuple[int, str, str]],
                  subagents: list[tuple[str, list[dict]]]) -> list[tuple[str, int, str, str]]:
    """一个 session 切成 turn，返回 (created_at, prompt_number, prompt 原文, 正文)；正文是空的 turn 不要。
    subagents 是 (第一行的 timestamp, 各行)，整份算进第一行所在的 turn，接在主 transcript 的内容后面"""
    starts = find_starts(rows, prompts)
    out = []
    for k, (i, n, at, text) in enumerate(starts):
        end, end_at = (starts[k + 1][0], starts[k + 1][2]) if k + 1 < len(starts) else (len(rows), "9999")
        parts = [p for r in rows[i:end] if in_window(r) for p in pieces(r)]
        for first, srows in subagents:
            if at <= first < end_at:
                parts += [p for r in srows if in_window(r) for p in pieces(r)]
        body = "\n".join(p for p in parts if p.strip())
        if body:
            out.append((at, n, text, body))
    return out


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def read_turns(src: str, transcripts: str) -> list[dict]:
    """claude-mem 记过的 session，有 transcript 的，切成 turn；按 prompt 的时间、session、prompt_number 排，从 FIRST_TURN_ID 起编 id"""
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    found = []
    for (cs,) in conn.execute(
        "SELECT DISTINCT content_session_id FROM user_prompts WHERE created_at <= ? ORDER BY content_session_id",
        (CUT,),
    ).fetchall():
        files = glob.glob(f"{transcripts}/*/{cs}.jsonl")
        if not files:
            continue
        rows = read_jsonl(files[0])
        if not any(in_window(r) for r in rows):
            continue
        prompts = conn.execute(
            "SELECT prompt_number, prompt_text, created_at FROM user_prompts "
            "WHERE content_session_id = ? ORDER BY prompt_number",
            (cs,),
        ).fetchall()
        base = files[0][:-len(".jsonl")]
        subagents = []
        for p in sorted(glob.glob(f"{base}/subagents/*.jsonl")
                        + glob.glob(f"{base}/subagents/workflows/*/agent-*.jsonl")):
            srows = [r for r in read_jsonl(p) if "timestamp" in r]
            if srows:
                subagents.append((min(r["timestamp"] for r in srows), srows))
        subagents.sort(key=lambda s: s[0])
        found += [(at, cs, n, text, body) for at, n, text, body in session_turns(rows, prompts, subagents)]
    conn.close()
    found.sort(key=lambda t: t[:3])
    return [
        {"id": FIRST_TURN_ID + j, "session_id": cs, "project": "", "type": "turn",
         "title": norm(text)[:60], "subtitle": "", "narrative": body,
         "facts": [], "files": [], "created_at": at}
        for j, (at, cs, n, text, body) in enumerate(found)
    ]


def window_observation_ids(src: str, transcripts: str) -> set[int]:
    """A 边：id ≤ UPTO、created_at ≥ T0 的 observation，去掉窗口里没有 transcript 的 session 压出来的"""
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    missing = [cs for (cs,) in conn.execute(
        "SELECT DISTINCT content_session_id FROM user_prompts WHERE created_at >= ? AND created_at <= ?",
        (T0, CUT),
    ).fetchall() if not glob.glob(f"{transcripts}/*/{cs}.jsonl")]
    dropped = {i for cs in missing for (i,) in conn.execute(
        "SELECT o.id FROM observations o JOIN sdk_sessions s ON o.memory_session_id = s.memory_session_id "
        "WHERE s.content_session_id = ? AND o.id <= ? AND o.created_at >= ?",
        (cs, UPTO, T0),
    )}
    ids = {i for (i,) in conn.execute(
        "SELECT id FROM observations WHERE id <= ? AND created_at >= ?", (UPTO, T0))}
    conn.close()
    return ids - dropped


def main(config_path: str) -> None:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    if cfg["unit"] == "observation":
        if Path(cfg["db"]).exists():
            sys.exit(f"{cfg['db']} 已经存在；要重拷先删掉它")
        mem = core.Memory(cfg)
        mem.store.copy_from(OBS_DB, window_observation_ids(SRC, TRANSCRIPTS))
    else:
        turns = read_turns(SRC, TRANSCRIPTS)
        mem = core.Memory(cfg)
        _, last = mem.store.stats()  # 断了重跑时，从库里最大的 id 后面接着灌
        todo = [t for t in turns if last is None or t["id"] > last]
        done = len(turns) - len(todo)
        for i in range(0, len(todo), BATCH):
            mem.add(todo[i:i + BATCH])
            print(f"已灌 {done + min(i + BATCH, len(todo))}/{len(turns)}", flush=True)
    print(mem.store.stats())


if __name__ == "__main__":
    main(sys.argv[1])
