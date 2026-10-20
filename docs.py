"""repo 里的文档：目录在哪、怎么切节、Status 行、repo 根。只有这个文件知道文档长什么样（和 transcript.py 对称）"""
import glob
import os
import re
import subprocess
import time

DIRS = (".claude/adr", "docs/adr", ".claude/specs", ".claude/design")
GLOBS = ("docs/*.md",)
SECTION = 2000  # 一节最多这么多字符，超了按空行再切
SUPERSEDED = re.compile(r"(?im)^status:.*superseded")
HEADING = re.compile(r"^(#{1,3}) +(.*\S)")
PREFACE = "前言"


def repo_root(cwd: str) -> str | None:
    """cwd 所在 repo 的主 checkout 根（worktree 归主 checkout）；不是 repo、git 出错、或根下一份文档都没有就 None"""
    try:
        r = subprocess.run(["git", "-C", cwd, "rev-parse", "--git-common-dir"],
                           capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    root = os.path.dirname(os.path.abspath(os.path.join(cwd, r.stdout.strip())))
    return root if files(root) else None


def files(root: str) -> list[str]:
    """要进库的文档，绝对路径排好序；Status 行含 superseded 的不要"""
    paths = set()
    for d in DIRS:
        paths.update(glob.glob(os.path.join(root, d, "*.md")))
    for g in GLOBS:
        paths.update(glob.glob(os.path.join(root, g)))
    keep = []
    for p in paths:
        with open(p, errors="replace") as f:
            if not SUPERSEDED.search(f.read(4000)):
                keep.append(p)
    return sorted(keep)


def mtimes(root: str) -> dict[str, int]:
    return {p: os.stat(p).st_mtime_ns for p in files(root)}


def stale(root: str, known: dict[str, int]) -> bool:
    return mtimes(root) != known


def split(body: str) -> list[str]:
    out = []
    while len(body) > SECTION:
        cut = body.rfind("\n\n", 0, SECTION)
        if cut < SECTION // 4:
            cut = SECTION
        out.append(body[:cut].strip())
        body = body[cut:].strip()
    return out + [body]


def sections(path: str, root: str) -> list[dict]:
    """一份文档切成节：#、##、### 起一节（fenced code 里的不算），第一个标题前的算「前言」，#### 以下归上一节，空的和只有标题行的节丢掉"""
    rel = os.path.relpath(path, root)
    with open(path, errors="replace") as f:
        text = f.read()
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(path)))
    out, buf, title, fenced = [], [], PREFACE, False

    def flush():
        body = "\n".join(buf).strip()
        if body and not ("\n" not in body and HEADING.match(body)):  # 只有一行标题的节不要（比如 spec 的 H1 紧跟着 ## 一）
            out.extend({"session_id": "", "project": os.path.basename(root), "type": "doc",
                        "title": f"{rel} § {title}", "subtitle": "", "narrative": piece,
                        "facts": [], "files": [rel], "created_at": created} for piece in split(body))
        buf.clear()

    for line in text.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        m = None if fenced else HEADING.match(line)
        if m:
            flush()
            title = m.group(2).strip()[:80]
        buf.append(line)
    flush()
    return out
