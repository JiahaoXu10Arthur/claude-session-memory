# claude-session-memory

English | [简体中文](README.zh-CN.md)

Cross-session memory for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Before each prompt, it searches records of earlier sessions and injects the most relevant ones into the context. It starts as a re-implementation of [claude-mem](https://github.com/thedotmack/claude-mem), an open-source memory plugin for Claude Code, then changes one variable at a time and measures each change on the same ten questions as the row before it. Python, SQLite, local models, no API keys.

**The baseline is a ruler: every later row changes one variable against it.**

Each question asks how something was solved in an earlier session, and the answer is the record of that earlier work. **hit@5** is how many of the ten answers rank in the top 5. **MRR** (mean reciprocal rank) is the average of 1/rank, so 1.0 means every answer ranked first.

| Question set | From | To | hit@5 | MRR |
|---|---|---|---|---|
| first ten questions | baseline (claude-mem's approach: dense search only, all-MiniLM-L6-v2) | experiment 2 (+ BM25, bge-m3) | 2/10 → 6/10 | 0.133 → 0.547 |
| second ten questions | experiment 3 · A (larger corpus) | experiment 5 (+ dedup, rerank) | 8/10 → 9/10 | 0.585 → 0.671 |

> **About the data.** The evaluation runs on the author's local Claude Code transcripts and the SQLite databases built from them. They are not published, so the numbers below can only be reproduced on the author's machine. Three questions in `eval.py` had project names removed for publication, so their wording differs slightly from the queries that produced the numbers.

## Terms

- **Turn**: one user prompt plus everything the agent did until it stopped.
- **Observation**: one memory record compressed from a turn, with a title, subtitle, narrative, facts and files.
- **Dense search**: nearest-neighbour search over embeddings. **BM25**: keyword search, here SQLite FTS5.
- **RRF** (reciprocal rank fusion): merges the dense and BM25 rankings by summing 1/(60 + rank).
- **Rerank**: a cross-encoder re-scores the fused top 20 against the query.

## How it works

```
write path   Stop hook ──POST /write──> serve.py queue ──worker──> transcript.next_turn() ──> core.write(): LLM compression ──> store.write()
read path    prompt ──UserPromptSubmit hook──> serve.py /search ──> dense + BM25 ──> RRF ──> rerank top 20 ──> top 5 into context
docs lookup  same /search, one more source: the repo's own ADR / spec / design files, split by heading ──> top 3 sections into context
```

- **Write path.** After each answer, the Stop hook sends the transcript path to a long-running local service. The service cuts the next turn out of the transcript (tool output truncated to 2,000 characters), compresses it with a local `qwen3.5:9b` into observations, stores them in SQLite and moves a byte cursor past that turn. One worker compresses one turn at a time, newest hook request first. Every 10 minutes a sweep adds recently changed transcripts to the back of the queue.
- **Read path.** When a prompt is sent, the hook uses it as the query. The service runs dense search (bge-m3) and BM25, fuses them with RRF, reranks the top 20 with bge-reranker-v2-m3, drops records from the current session, and returns the top 5 as context. One search takes 2–3 seconds.
- **Docs lookup.** The same request also searches the current repo's own documents (ADRs, specs, design docs, split by heading) and appends the top 3 sections after the memory results. These sections are not reranked, and a section already injected in the session is not injected again. Each repo gets its own SQLite file, built in the background and updated when a file's mtime changes.

Each file owns one boundary. `core.py` never mentions Claude, SQLite or bge.

| File | Responsibility | Edit it to swap… |
|---|---|---|
| `core.py` | compression prompt, search pipeline, fusion | the retrieval logic |
| `store.py` | SQLite + sqlite-vec + FTS5 | the vector store |
| `embed.py` | embedding and rerank models | the retrieval models |
| `llm.py` | request to the compression model (Ollama's OpenAI-compatible `/v1`) | the compression model |
| `transcript.py` | cutting Claude Code transcripts into turns | the transcript format |
| `docs.py` | finding and splitting repo documents | the docs layout |
| `serve.py` | HTTP service, write queue, worker, sweep | the process model |
| `hooks/` | Claude Code hooks (`prompt.py`, `stop.py`) | the client |

## Results

Up to 6 · A, the corpus is observations that claude-mem had already written, imported from its database. Only 6 · B uses records written by this project's write path. The comparison restarts twice: 3 · A switches to a larger corpus and a new question set, and experiment 6 has its own sessions and questions, so 6 · A and 6 · B are only comparable with each other.

| Exp. | The one thing changed | hit@5 | MRR |
|---|---|---|---|
| baseline | claude-mem's approach: dense search only, all-MiniLM-L6-v2 | 2/10 | 0.133 |
| 1 | add BM25, RRF k=60 | 3/10 | 0.268 |
| 2 | embedding → bge-m3 | 6/10 | 0.547 |
| 3 · A | new corpus (15,479 observations, 2026-08-26 to 09-24), ten new questions; any observation from the answer turn counts | 8/10 | 0.585 |
| 3 · B | no compression: index the transcript text (tool output cut to 2,000 characters), each turn split into 512-token chunks | 4/10 | 0.318 |
| 4 | dedup: merge observations from the same session and day with similarity > 0.9 (15,222 left) | 8/10 | 0.598 |
| 5 | add rerank: bge-reranker-v2-m3 over the fused top 20 | 9/10 | 0.671 |
| 5 · ext | experiment 5 without BM25: dense top 20, then rerank | 9/10 | 0.676 |
| 6 · A | 9 sessions from 2026-09-24 to 09-27 on top of the experiment 4 corpus, records written by claude-mem (claude-sonnet-4-6) | 8/10 | 0.582 |
| 6 · B | same sessions, records written by this project's write path (qwen3.5:9b, tool output truncated to 2,000 characters) | 9/10 | 0.700 |
| 7 | time-weighted ranking in the read path, two variants | rejected | no net gain across five groups |

What each row shows:

- **1 → 2.** The jump from 3/10 to 6/10 comes from the embedding, not from BM25. The corpus mixes Chinese and English, and the 384-dimensional MiniLM was trained on English only. BM25 alone already scored 6/10 in experiment 1, and fusion was worse than BM25 alone at that point. In experiment 2, fusion (6/10) still lost to dense alone (8/10). On the experiment 4 corpus the three were level: fusion 8/10 · 0.598, dense only 7/10 · 0.588, BM25 only 7/10 · 0.586.
- **3 · A vs 3 · B.** Compression is worth it. On the same transcripts, indexing compressed observations scores 8/10, and indexing turn text directly scores 4/10. Tool output is over 70% of the transcript text and buries the answers.
- **4.** Dedup removed 257 observations. Ranks barely moved. The benefit is fewer duplicates in the injected context.
- **5.** Rerank is the biggest single step in the read path: MRR from 0.598 to 0.671. Reranking the full text of 20 candidates adds about 4 seconds per search. Truncating each candidate to 1,500 characters first halves that time without lowering ranks.
- **5 · ext.** With rerank on, fusion adds nothing: dense + rerank scores 0.676 against fusion + rerank 0.671. None of the ten answers fell outside the dense top 20, so BM25 is redundant on this question set and is kept only as a fallback.
- **6.** This compares the whole write path. A local 9B model with truncated tool output produced records that are easier to find than the ones claude-mem writes with Sonnet. Its top-1 hit is more often the event itself rather than a progress note ("handoff written", "committed"). It also invents things: once it recorded a "Removed reasoning_effort" change that never happened in the code. Its cap of 8 observations per compression call is often reached.
- **7.** In experiment 6, an old conclusion once ranked above the newer one that overturned it. Neither time-weighting variant fixed that case, and both lowered other results, so neither was adopted.

## Running it

Requirements: [uv](https://docs.astral.sh/uv/) and Python ≥ 3.14. The write path also needs [Ollama](https://ollama.com) with `qwen3.5:9b` and a server-side `context_length` of 32768. The service runs with `HF_HUB_OFFLINE=1`, so download the models once beforehand (the `python -c` line below).

```bash
uv sync
mkdir -p data                                       # databases and serve.log go here
uv run python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"
uv run pytest tests -q                              # 172 tests
uv run python serve.py configs/live.toml            # the long-running service
```

`eval.py configs/<experiment>.toml` (second question set, experiments 3–5) and `eval_exp6.py` reproduce the tables, but they need the author's databases (see About the data). The first question set is not in the repo.

To use it in Claude Code, register the prompt hook in `.claude/settings.local.json` of the project where you want memory. If the service is not running, the hook starts it, and that prompt gets no injection.

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command",
        "command": "uv run --directory /path/to/claude-session-memory python hooks/prompt.py configs/live.toml",
        "timeout": 15 } ] }
    ]
  }
}
```

A fresh `data/live.db` is empty, so at first only docs lookup returns anything. Memory fills up once the write path is on: register `hooks/stop.py` the same way as a Stop hook, and add `[write] sweep = true` to `configs/live.toml` to turn on the sweep. The author keeps the write path off outside experiments, because the sweep starts Ollama, which takes 6.7 GB. Turns before 2026-09-24 are skipped, and the sweep does not backfill sessions that started before 2026-09-26 (`CUT` and `LAUNCH` in `serve.py`). The service exits after 30 minutes without a `/search` or `/write` request, once the worker is idle.

## Known limitations

- The evaluation set is ten hand-written questions, all from the author's own projects. The numbers can tell which steps are worth taking. They cannot resolve a difference of 0.02.
- From 3 · A on, answers are judged by turn. Rows 3 · A, 4, 5, 5 · ext and 6 · A use existing records, and the `prompt_number` that maps each record to a turn does not always point to the prompt the content came from. This error is accepted.
- Write path: sessions from `claude -p` and older claude-desktop versions have no summary line and are skipped. If a session is killed mid-turn, the last segment is not compressed. When a turn quotes an earlier record, the 9B model records it as done in this turn. A turn is compressed in at most 8 windows of 40,000 characters each, and the rest is dropped.
- Read path: prompts shorter than 15 characters get no injection. Every longer prompt gets 5 results, even when it has nothing to do with past work. BM25 cannot help when a Chinese query has to match an English record (去重 vs. deduplication).
- Docs lookup: every prompt that reaches the service gets 3 sections, even when it has nothing to do with the docs. The first prompt in a new repo gets no docs. On a cold start, the first prompt can exceed the hook's 12-second limit and gets nothing.

## License

[MIT](LICENSE)
