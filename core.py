import json
import sys
import threading

import numpy as np

import embed
import llm
import store

RRF_K = 60
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
RERANK_TOP = 20
TITLE_CHARS = 60

TYPES = ["bugfix", "feature", "refactor", "change", "discovery", "decision",
         "security_alert", "security_note", "sensitive"]

SCHEMA = {
    "type": "object",
    "properties": {"observations": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"}, "subtitle": {"type": "string"}, "narrative": {"type": "string"},
            "facts": {"type": "array", "items": {"type": "string"}},
            "files": {"type": "array", "items": {"type": "string"}},
            "type": {"type": "string", "enum": TYPES},
        },
        "required": ["title", "subtitle", "narrative", "facts", "files", "type"],
        "additionalProperties": False,
    }}},
    "required": ["observations"],
    "additionalProperties": False,
}

PROMPT = """Below is the transcript of one finished turn of a coding-agent session. Turn it into memory records that a later session can search.

Each record states a result of that turn: something learned, built, fixed, configured or decided. Never describe your own reading or writing.
Everything you need is in the transcript; you cannot run tools and do not need to.

WHAT TO KEEP
------------
Record things that will still be true and useful later:
- Behaviour the system has after this turn that it did not have before
- Fixes, features, config and docs that were finished
- Findings backed by evidence: log lines, traces, database rows, measurements, code that was read
- Decisions the user made, with the reason given

Write each record as the outcome itself, with an active verb such as added, fixed, removed, moved, measured, found, chose.

Good: "Search results are reranked by a cross-encoder before injection"
Good: "The hook timed out because loading the embedding model took 14 seconds"
Bad (describes the act of recording): "Read through serve.py and wrote down observations"

WHAT TO LEAVE OUT
-----------------
Skip what carries no new information: status checks that found nothing, installs that succeeded without incident, directory listings that led nowhere, a repeat of something already recorded, a search that returned nothing.
Turns that only ask about progress or status ("is it done", "how much is left", "is the GPU busy"): skip.
Edits to handoff, ledger, memory, notes, or progress files: skip unless the turn also records a new decision.
A turn that explains or walks through one concept: at most ONE observation for it, not one per fact.
If the whole turn is routine, return {"observations": []}.

FIELDS
------
title: a few words naming what happened; name the mechanism or identifier involved (e.g. reasoning_effort, filter-branch), not just the area
subtitle: one sentence, at most 24 words, saying what the title means
narrative: the full account: the change or finding, the mechanism behind it, and its consequence
facts: short statements that each make sense alone; one piece of information per fact, names instead of pronouns, concrete file names, functions and values
files: every file the turn read or changed, as paths from the project root
type: exactly one of
  - bugfix: broken behaviour that now works
  - feature: a capability that did not exist before
  - refactor: structure changed, behaviour the same
  - change: any other edit, such as docs or config
  - discovery: something learned about how the existing system works
  - decision: a design choice, with the reason for it
  - security_alert: a security problem to deal with before going on
  - security_note: a security detail worth keeping that needs no action now
  - sensitive: details that should stay in this context, such as internal hostnames, plans not yet announced, people's private details, revenue figures, customer names

One observation per distinct thing learned, built, fixed or decided - usually 2 to 8 per turn; do not merge unrelated findings into one.
Write in English; keep identifiers, paths, commands and quoted terms exactly as they appear. Output one JSON object {"observations": [...]} and nothing else.

TRANSCRIPT
----------
"""
PROMPT_TAIL = """

----------
Now output the JSON object: up to 8 observations from the transcript above, in English; an empty list if the turn is routine."""


def chunks(obs: dict, embedder: embed.Embedder) -> list[str]:
    """observation：narrative 一条、每个 fact 一条，和上游一样。turn：narrative 按 token 切，
    每块 CHUNK_SIZE 个、相邻两块重叠 CHUNK_OVERLAP 个。title、subtitle 都不进向量"""
    if obs["type"] == "turn":
        return embedder.split_text(obs["narrative"], CHUNK_SIZE, CHUNK_OVERLAP)
    return [t for t in [obs["narrative"], *obs["facts"]] if t]


def rerank_text(obs: dict, limit: int | None = None) -> str:
    """给 reranker 看的文本：observation 的 narrative 和每个 fact 换行拼起来，和它进向量的文字一样，title 不放；
    limit 给了就只留前 limit 个字符"""
    return "\n".join(t for t in [obs["narrative"], *obs["facts"]] if t)[:limit]


def format(hits: list[dict]) -> str:
    """注进 context 的那段：一行标题，每条一行 title 加 subtitle（subtitle 是上游写的一句话说明，空的只印 title）"""
    lines = [f"memory：和这句话相关的 {len(hits)} 条"]
    for h in hits:
        lines.append(f"- [{h['type']} · {h['project']} · {h['created_at'][:10]}] {h['title']}"
                     + (f" — {h['subtitle']}" if h["subtitle"] else ""))
    return "\n".join(lines)


DOC_CHARS = 400
INDEX_BATCH = 16  # 建文档索引时每拿一次锁 embed 这么多节


def format_docs(hits: list[dict]) -> str:
    """docs 那段：一行标题，每节两行，title 一行、正文（空白压成一个空格）前 DOC_CHARS 字一行"""
    lines = [f"docs：和这句话相关的 {len(hits)} 节"]
    for h in hits:
        lines.append(f"- [{h['title']}]")
        lines.append("  " + " ".join(h["narrative"].split())[:DOC_CHARS])
    return "\n".join(lines)


def rrf(rankings: list[list[int]], k: int) -> list[int]:
    """每个 id 的分数是它在各路的 1/(k + 名次) 之和；分数高的在前，同分时先出现的在前"""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for r, obs_id in enumerate(ranking, 1):
            scores[obs_id] = scores.get(obs_id, 0.0) + 1 / (k + r)
    return sorted(scores, key=lambda obs_id: -scores[obs_id])


class Memory:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.embedder = embed.Embedder(cfg["embed"]["model"])
        self.store = store.Store(cfg["db"], self.embedder.dim)
        self.reranker = embed.Reranker(cfg["search"]["rerank"]) if "rerank" in cfg["search"] else None
        self.llm = llm.LLM(cfg["llm"]) if "llm" in cfg else None
        self.lock = threading.Lock()  # 模型和库一次只让一个线程用；write 里模型调用在锁外

    def write(self, text: str, meta: dict) -> list[dict]:
        """把一段 turn 正文压成 observation 存进库，返回存进去的（带 id）。meta：session_id、start、project、
        created_at、prompt、cursor=(path, offset) 或 None（不是这个 turn 的最后一段时不挪游标）。
        模型回的东西不成形就记 0 条，turn 正文和游标照写；模型连不上就抛出去，什么都不写"""
        raw = self.llm.complete(PROMPT + text + PROMPT_TAIL, SCHEMA)
        try:
            observations = [{"session_id": meta["session_id"], "project": meta["project"], "created_at": meta["created_at"],
                             **{k: it[k] for k in ("type", "title", "subtitle", "narrative", "facts", "files")}}
                            for it in json.loads(raw)["observations"]]
        except (ValueError, KeyError, TypeError) as e:
            print(f"write: {meta['session_id']} {meta['start']} 模型输出不成形 {e!r}: {str(raw)[:200]!r}", file=sys.stderr, flush=True)
            observations = []
        turn = {"session_id": meta["session_id"], "start": meta["start"], "project": meta["project"],
                "created_at": meta["created_at"], "title": meta["prompt"][:TITLE_CHARS], "text": text}
        with self.lock:
            vecs = [self.embedder.embed(chunks(o, self.embedder)) for o in observations]
            return self.store.write(turn, observations, vecs, meta["cursor"])

    def add(self, observations: list[dict]) -> None:
        chunks_per_obs = [chunks(o, self.embedder) for o in observations]
        texts = [c for cs in chunks_per_obs for c in cs]
        vecs = self.embedder.embed(texts)
        batch = []
        i = 0
        for o, cs in zip(observations, chunks_per_obs):
            n = len(cs)
            batch.append((o, vecs[i:i + n]))
            i += n
        self.store.add(batch)

    def index(self, store, prefix: str, records: list[dict]) -> None:
        """一份文档的节：每 INDEX_BATCH 条拿一次锁 embed（一份大文档不能把锁占几十秒，search 要插进来），
        最后锁里让 store 把 title 以 prefix 开头的旧节换成这批"""
        batch = []
        for i in range(0, len(records), INDEX_BATCH):
            with self.lock:
                batch += [(o, self.embedder.embed(chunks(o, self.embedder))) for o in records[i:i + INDEX_BATCH]]
        with self.lock:
            store.replace_file(prefix, batch)

    def rankings(self, query: str, k: int, store=None) -> dict[str, list[int]]:
        store = store or self.store
        mode = self.cfg["search"]["mode"]
        if mode not in ("dense", "hybrid"):
            raise ValueError(mode)
        vec = self.embedder.embed([query])[0]
        out = {"dense": [obs_id for obs_id, _ in store.search_dense(vec, k)]}
        if mode == "hybrid":
            out["bm25"] = [obs_id for obs_id, _ in store.search_bm25(query, k)]
        return out

    def search(self, query: str, k: int, store=None, rerank: bool = True) -> list[dict]:
        """融合后的前 k 条。store 给了就搜那个库；配了 rerank 且 rerank=True 就把前 RERANK_TOP 条按 reranker 的分重排，后面的不动"""
        store = store or self.store
        with self.lock:
            hits = store.get(rrf(list(self.rankings(query, k, store).values()), RRF_K)[:k])
            if self.reranker is None or not rerank:
                return hits
            head = hits[:RERANK_TOP]
            limit = self.cfg["search"].get("rerank_chars")
            scores = self.reranker.score(query, [rerank_text(o, limit) for o in head])
        return [head[i] for i in np.argsort(-scores, kind="stable")] + hits[RERANK_TOP:]
