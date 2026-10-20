"""Claude Code 的 transcript：切 turn、字节游标、subagent 归属、扫尾要读的文件。只有这个文件知道 transcript 的格式"""
import glob
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

EXCLUDE = ("claude-mem-observer-sessions",)
TOOL_OUTPUT = 2000  # 每个 tool 输出只留前 2000 字，和实验 3 的 B 边一样
WINDOW = 40_000  # 一次压缩最多吃这么多字符
MAX_WINDOWS = 8  # 一个 turn 最多压这么多段，剩下的丢掉
PROMPT_HEAD = 300  # 续篇、subagent、第 2 段起，正文前面加这个 turn 的 prompt 前 300 字
SUBAGENT_IDLE = 300  # subagent 文件 5 分钟没改过才算写完


@dataclass
class Turn:
    session_id: str
    start: int  # 起点行的字节偏移，(session_id, start) 是 turn 的键
    created_at: str
    project: str
    prompt: str  # 起点那句 prompt 的原文
    text: str
    end: int  # 压完把游标放这里
    continuation: bool  # 续篇和 subagent：正文里没有 prompt，压的时候每段前面都加 prompt 开头


def text_of(row: dict) -> str:
    c = row.get("message", {}).get("content")
    return c if isinstance(c, str) else "".join(b.get("text", "") for b in c or [] if b.get("type") == "text")


def is_start(row: dict) -> bool:
    """人打的 prompt：user 行 origin.kind 是 human（compact 出来的摘要不算）；排队送进去的 prompt；/goal 行"""
    if row.get("type") == "user":
        if row.get("origin", {}).get("kind") == "human" and not row.get("isCompactSummary"):
            return True
        return not row.get("isMeta") and text_of(row).startswith("<command-name>/goal</command-name>")
    a = row.get("attachment", {})
    return row.get("type") == "attachment" and a.get("type") == "queued_command" and a.get("commandMode") == "prompt"


def prompt_of(row: dict) -> str:
    if row.get("type") == "attachment":
        p = row["attachment"].get("prompt", "")
        return p if isinstance(p, str) else "".join(b.get("text", "") for b in p if isinstance(b, dict))
    return text_of(row)


def pieces(row: dict) -> list[str]:
    """一行里放进 turn 的几段。attachment 只放排队送进去的 prompt；isMeta 的行和图片不放；tool 输出截短"""
    if row.get("type") == "attachment":
        return [prompt_of(row)] if is_start(row) else []
    if row.get("isMeta") or row.get("isCompactSummary"):  # compact 摘要讲的是旧 turn 的事，不再压一遍
        return []
    c = row.get("message", {}).get("content")
    out = []
    if row.get("type") == "user":
        if isinstance(c, str):
            return [c]
        for b in c or []:
            if b.get("type") == "tool_result":
                r = b.get("content")
                if isinstance(r, list):
                    r = [x for x in r if x.get("type") != "image"]
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


def project(rs: list[tuple[int, int, dict]]) -> str:
    """session 开头的 cwd：worktree 记 <repo>/<worktree>，其余记目录名。中途 cd 过的行不看"""
    cwd = next((r["cwd"] for _, _, r in rs if "cwd" in r), "")
    parts = Path(cwd).parts
    if ".claude" in parts and parts.index(".claude") + 2 < len(parts) and parts[parts.index(".claude") + 1] == "worktrees":
        i = parts.index(".claude")
        return f"{parts[i - 1]}/{parts[i + 2]}"
    return Path(cwd).name


def rows(path: str, session_id: str) -> list[tuple[int, int, dict]]:
    """(这行的字节偏移, 下一行的字节偏移, 行)。只要完整的行（末尾有换行）和 sessionId 对得上的行；
    fork 出来的 session 带着父 session 的行，不要"""
    out = []
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    for line in data.splitlines(keepends=True):
        if line.endswith(b"\n"):
            row = json.loads(line)
            if row.get("sessionId") == session_id:
                out.append((offset, offset + len(line), row))
        offset += len(line)
    return out


def body(rs: list[dict]) -> str:
    return "\n".join(p for r in rs for p in pieces(r) if p.strip())


def session_of(path: str) -> str:
    return Path(path).stem


def parent_of(path: str) -> str:
    """subagent 文件 → 它父 session 的主文件"""
    p = Path(path)
    while p.name != "subagents":
        p = p.parent
    return str(p.parent) + ".jsonl"


def started_at(path: str) -> str | None:
    """这个 session 自己第一行的 timestamp（fork 出来的文件可能一行都没有）；
    第 4 步用它判断一份没游标的文件是上线前就有的还是新建的"""
    sid = session_of(parent_of(path)) if "/subagents/" in path else session_of(path)
    return next((r["timestamp"] for _, _, r in rows(path, sid) if "timestamp" in r), None)


def next_turn(path: str, cursor: int) -> Turn | int:
    """主文件：游标起、到最后一行 stop_hook_summary 为止的第一段有正文的 turn。游标不在起点上的那段算上一个 turn 的续篇，
    键和 prompt 用上一个起点的。summary 之后的行下次再说。
    没有可压的就返回一个偏移：游标可以直接挪到那里（一行 summary 都没有、或游标已经过了最后一个 summary，就是原来的游标）"""
    sid = session_of(path)
    rs = rows(path, sid)
    ends = [e for _, e, r in rs if r.get("subtype") == "stop_hook_summary"]
    if not ends:
        return cursor
    limit = ends[-1]
    starts = [(o, r) for o, _, r in rs if is_start(r)]
    proj = project(rs)
    while cursor < limit:
        before = [(o, r) for o, r in starts if o <= cursor]
        after = [o for o, _ in starts if o > cursor]
        end = min([*after, limit])
        if not before:  # 第一个起点之前的行（session 开头的环境信息）
            cursor = end
            continue
        so, sr = before[-1]
        segment = [r for o, _, r in rs if cursor <= o < end]
        text = body(segment)
        if not text:
            cursor = end
            continue
        return Turn(sid, so, sr["timestamp"], proj, prompt_of(sr), text, end, so < cursor)
    return cursor


def spawned_at(path: str, prs: list[tuple[int, int, dict]]) -> int | None:
    """subagent 旁边的 .meta.json 记着派它的那次 tool_use 的 id，找到这条 tool_use 在父文件里的偏移；
    workflow 的 agent 和嵌套的 subagent 找不到，返回 None"""
    meta = Path(path).with_suffix(".meta.json")
    if not meta.exists():
        return None
    tool_use = json.loads(meta.read_text()).get("toolUseId")
    for o, _, r in prs:
        if r.get("type") == "assistant" and any(
                b.get("type") == "tool_use" and b.get("id") == tool_use for b in r["message"]["content"]):
            return o
    return None


def next_subagent(path: str, cursor: int) -> Turn | int:
    """subagent 文件没有 summary 行，整份压：父文件最后一个 summary 晚于它最后一行、文件 5 分钟没改过才算写完，
    没写完就返回原游标。归到派它的那次 tool_use 所在的 turn；meta 里查不到就归父 session 起点时间不晚于它
    第一行的最后一个 turn。父文件一个起点都没有（claude -p 的 session）就不压，返回文件末尾"""
    parent = parent_of(path)
    sid = session_of(parent)
    rs = [(o, e, r) for o, e, r in rows(path, sid) if "timestamp" in r]
    if not rs or time.time() - os.path.getmtime(path) < SUBAGENT_IDLE:
        return cursor
    prs = rows(parent, sid)
    summaries = [r["timestamp"] for _, _, r in prs if r.get("subtype") == "stop_hook_summary"]
    if not summaries or summaries[-1] <= rs[-1][2]["timestamp"]:
        return cursor
    end = rs[-1][1]
    starts = [(o, r) for o, _, r in prs if is_start(r)]
    if not starts:
        return end
    at = spawned_at(path, prs)
    if at is None:
        first = rs[0][2]["timestamp"]
        before = [s for s in starts if s[1]["timestamp"] <= first]
    else:
        before = [s for s in starts if s[0] <= at]
    so, sr = before[-1]
    text = body([r for o, _, r in rs if o >= cursor])
    if not text:
        return end
    return Turn(sid, so, sr["timestamp"], project(prs), prompt_of(sr), text, end, True)


def windows(turn: Turn) -> list[str]:
    """按 WINDOW 切，最多 MAX_WINDOWS 段；第 2 段起前面加 prompt 开头，续篇和 subagent 第 1 段也加"""
    head = turn.prompt[:PROMPT_HEAD] + "\n"
    out = [turn.text[i:i + WINDOW] for i in range(0, len(turn.text), WINDOW)][:MAX_WINDOWS]
    skip = 0 if turn.continuation else 1
    return out[:skip] + [head + w for w in out[skip:]]


def sweep(root: str = "~/.claude/projects", days: int = 2) -> list[str]:
    """最近 days 天改过的主文件和 subagent 文件，新改的在前；claude-mem observer 的不要"""
    root = os.path.expanduser(root)
    since = time.time() - days * 86400
    paths = (glob.glob(f"{root}/*/*.jsonl") + glob.glob(f"{root}/*/*/subagents/*.jsonl")
             + glob.glob(f"{root}/*/*/subagents/workflows/*/agent-*.jsonl"))
    recent = [(os.path.getmtime(p), p) for p in paths if not any(x in p for x in EXCLUDE)]
    return [p for m, p in sorted(recent, reverse=True) if m >= since]
