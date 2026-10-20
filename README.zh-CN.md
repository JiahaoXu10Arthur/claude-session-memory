# claude-session-memory

[English](README.md) | 简体中文

给 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 做的跨 session 记忆。每句 prompt 发出去之前，先搜之前 session 的记录，把最相关的几条注入 context。先照 [claude-mem](https://github.com/thedotmack/claude-mem)（一个开源的 Claude Code 记忆插件）的做法复刻一遍，然后每次只改一个变量，每改一处都用和上一行相同的十道题量一次。Python，SQLite，本地模型，不需要任何 API key。

**The baseline is a ruler: every later row changes one variable against it.**
baseline 是量尺，后面每一行只改一个变量，数字才能互相比。

每道题问的都是「之前某个 session 里这件事是怎么解决的」，答案是记下那次工作的记录。**hit@5** 是十道题里答案排进前 5 的题数。**MRR**（mean reciprocal rank）是答案名次倒数的平均，1.0 表示每道题的答案都排第一。

| 题集 | 起点 | 终点 | hit@5 | MRR |
|---|---|---|---|---|
| 第一套十道题 | baseline（照 claude-mem：只用 dense 检索，all-MiniLM-L6-v2） | 实验 2（加 BM25，换 bge-m3） | 2/10 → 6/10 | 0.133 → 0.547 |
| 第二套十道题 | 实验 3 · A（换成更大的语料） | 实验 5（加去重和 rerank） | 8/10 → 9/10 | 0.585 → 0.671 |

> **关于数据。**eval 的原始数据是作者本机的 Claude Code transcript 和从它压出的 SQLite 库，不随 repo 发布，所以下面的数字只能在作者的机器上复现。`eval.py` 里有三道题在发布时去掉了项目名，题面和量数字时用的 query 不完全一样。

## 术语

- **turn**：用户的一句 prompt，加上 agent 为它做的所有事，直到停下来。
- **observation**：从一个 turn 压出的一条记忆记录，有 title、subtitle、narrative、facts、files 五个字段。
- **dense 检索**：在 embedding 上找最近邻。**BM25**：关键词检索，这里用的是 SQLite FTS5。
- **RRF**（reciprocal rank fusion）：把 dense 和 BM25 两份排名合成一份，每条记录的分数是各路 1/(60 + 名次) 之和。
- **rerank**：用 cross-encoder 对融合后的前 20 条和 query 重新打分。

## 工作原理

```
write path   Stop hook ──POST /write──> serve.py 队列 ──worker──> transcript.next_turn() ──> core.write(): LLM 压缩 ──> store.write()
read path    prompt ──UserPromptSubmit hook──> serve.py /search ──> dense + BM25 ──> RRF ──> rerank 前 20 ──> 前 5 条注入 context
docs lookup  同一个 /search 多搜一路：当前 repo 自己的 ADR / spec / design 文件按标题切节 ──> 前 3 节注入 context
```

- **write path（写入）。**每轮回答完，Stop hook 把 transcript 路径发给本机的常驻服务。服务从 transcript 里切出下一个 turn（tool 输出只留前 2000 字），用本地的 `qwen3.5:9b` 压成若干条 observation，存进 SQLite，再把字节游标挪到这个 turn 后面。一个 worker 一次压一个 turn，hook 发来的最新请求先压。每 10 分钟定时扫描一遍最近改过的 transcript，放到队列末尾。
- **read path（读取）。**每句 prompt 发出去时，hook 拿这句话当 query 去问服务。服务做 dense 检索（bge-m3）和 BM25，用 RRF 融合，前 20 条用 bge-reranker-v2-m3 重排，去掉当前 session 自己的记录，前 5 条返回去当 context。一次搜索 2–3 秒。
- **docs lookup（搜文档）。**同一次请求还会搜当前 repo 自己的文档（ADR、spec、design 文档，按标题切节），前 3 节接在 memory 结果后面。这几节不做 rerank，同一个 session 里注入过的节不再注入。每个 repo 有自己的 SQLite 文件，在后台建，文件的 mtime 变了就更新。

一个文件管一个边界。`core.py` 里不出现 Claude、SQLite、bge 这些字眼。

| 文件 | 负责什么 | 想换掉什么就改它 |
|---|---|---|
| `core.py` | 压缩用的 prompt、搜索流程、融合 | 检索逻辑 |
| `store.py` | SQLite + sqlite-vec + FTS5 | 向量库 |
| `embed.py` | embedding 和 rerank 模型 | 检索用的模型 |
| `llm.py` | 请求压缩用的模型（Ollama 兼容 OpenAI 的 `/v1`） | 压缩用的模型 |
| `transcript.py` | 把 Claude Code 的 transcript 切成 turn | transcript 格式 |
| `docs.py` | 找到 repo 文档并切节 | 文档布局 |
| `serve.py` | HTTP 服务、写入队列、worker、定时扫描 | 进程模型 |
| `hooks/` | Claude Code 的 hook（`prompt.py`、`stop.py`） | 客户端 |

## 结果

6 · A 及之前的实验，语料都是 claude-mem 已经写好的 observation，从它的数据库导进来。只有 6 · B 用的是本项目 write path 写的记录。对比重新开始过两次：3 · A 换成更大的语料和一套新题，实验 6 有自己的 session 和题，所以 6 · A 和 6 · B 只能这两行之间比。

| 实验 | 只改了什么 | hit@5 | MRR |
|---|---|---|---|
| baseline | 照 claude-mem：只用 dense 检索，all-MiniLM-L6-v2 | 2/10 | 0.133 |
| 1 | 加 BM25，RRF k=60 | 3/10 | 0.268 |
| 2 | embedding 换成 bge-m3 | 6/10 | 0.547 |
| 3 · A | 语料换成 2026-08-26 到 09-24 的 15479 条 observation，题换成新的十道；答案 turn 压出的任何一条 observation 都算命中 | 8/10 | 0.585 |
| 3 · B | 不压缩：直接索引 transcript 正文（tool 输出截到 2000 字），每个 turn 按 512 token 切块 | 4/10 | 0.318 |
| 4 | 去重：同一 session、同一天、相似度 > 0.9 的 observation 合并，剩 15222 条 | 8/10 | 0.598 |
| 5 | 加 rerank：融合后的前 20 条用 bge-reranker-v2-m3 重排 | 9/10 | 0.671 |
| 5 · 补 | 实验 5 去掉 BM25：只取 dense 前 20 条再 rerank | 9/10 | 0.676 |
| 6 · A | 2026-09-24 到 09-27 的 9 个 session，加在实验 4 的语料上，记录由 claude-mem 写（claude-sonnet-4-6） | 8/10 | 0.582 |
| 6 · B | 同一批 session，记录由本项目的 write path 写（qwen3.5:9b，tool 输出截到 2000 字） | 9/10 | 0.700 |
| 7 | read path 按时间加权，两种做法 | 不采用 | 五组里没有净收益 |

每一行说明什么：

- **1 → 2。**从 3/10 到 6/10 靠的是换 embedding，BM25 没起作用。语料中英混杂，384 维的 MiniLM 只用英文训练过。实验 1 里只用 BM25 就有 6/10，当时融合还不如只用 BM25。实验 2 里融合 6/10，仍然输给只用 dense 的 8/10。实验 4 的语料上，三种持平：融合 8/10 · 0.598，只用 dense 7/10 · 0.588，只用 BM25 7/10 · 0.586。
- **3 · A 对 3 · B。**压缩值得做。同一批 transcript，压成 observation 再索引是 8/10，直接按 turn 切块索引正文是 4/10。transcript 正文里七成以上是 tool 输出，把答案淹没了。
- **4。**去重删了 257 条，名次几乎没变。好处是注入 context 时少几条重复。
- **5。**rerank 是 read path 里最大的一步，MRR 从 0.598 升到 0.671。20 条候选用全文做 rerank，每次搜索多花约 4 秒。每条候选先截到 1500 字再 rerank，这部分时间减半，名次不降。
- **5 · 补。**开了 rerank 之后，融合没有带来收益：只用 dense 加 rerank 是 0.676，融合加 rerank 是 0.671。十道题的答案都在 dense 前 20 条以内，所以在这十道题上 BM25 那一路是多余的，只留着当后备。
- **6。**这两行比的是整条 write path。本地 9B 模型加截短的 tool 输出，写出的记录比 claude-mem 用 Sonnet 写的更容易搜到，排第一的更常是那件事本身，而不是过程记录（「handoff 写好了」「已 commit」）。但它也会编造：有一次写出了一条代码里从没发生过的「Removed reasoning_effort」。每次压缩最多 8 条的上限也经常用满。
- **7。**实验 6 里有一次，旧结论排在了推翻它的新结论前面。两种按时间加权的做法都没修好这个例子，还拉低了别的结果，所以都不采用。

## 运行

需要 [uv](https://docs.astral.sh/uv/) 和 Python ≥ 3.14。write path 还需要 [Ollama](https://ollama.com) 和 `qwen3.5:9b`，Ollama 服务端的 `context_length` 要设成 32768。服务运行时设了 `HF_HUB_OFFLINE=1`，所以要先把模型下载好（下面 `python -c` 那一行）。

```bash
uv sync
mkdir -p data                                       # 数据库和 serve.log 放在这里
uv run python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"
uv run pytest tests -q                              # 172 个测试
uv run python serve.py configs/live.toml            # 常驻服务
```

`eval.py configs/<实验>.toml`（第二套题，实验 3–5）和 `eval_exp6.py` 能复现上面的表，但要用作者的数据库（见「关于数据」）。第一套题不在 repo 里。

要在 Claude Code 里用，在需要记忆的项目的 `.claude/settings.local.json` 里注册 prompt hook。服务没在运行时，hook 会把它启动起来，这一句 prompt 不注入。

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

新建的 `data/live.db` 是空的，一开始只有 docs lookup 有结果。打开 write path 之后记忆才会积累起来：照同样的写法把 `hooks/stop.py` 注册成 Stop hook，再在 `configs/live.toml` 里加 `[write] sweep = true` 打开定时扫描。作者只在做实验时打开 write path，因为定时扫描会启动 Ollama，占 6.7 GB。2026-09-24 之前的 turn 会跳过，定时扫描也不回灌 2026-09-26 之前开始的 session（`serve.py` 里的 `CUT` 和 `LAUNCH`）。服务 30 分钟没收到 `/search` 或 `/write` 请求、而且 worker 空闲时，就自己退出。

## 已知的不足

- 评测集只有十道人工出的题，全部来自作者自己的项目。这些数字能分出哪一步值得做，分不出 0.02 的差别。
- 从 3 · A 开始按 turn 判。3 · A、4、5、5 · 补、6 · A 用的是现成的记录，记录靠 `prompt_number` 对到 turn，这个号不一定是内容所在的那一句 prompt。这个误差接受了。
- write path：`claude -p` 和老版本 claude-desktop 的 session 没有 summary 行，不压。session 在一轮中间被 kill 时，最后一段不压。一轮里转述之前的记录时，9B 模型会记成本轮做的事。一个 turn 最多压 8 段、每段 40000 字符，超出的部分丢掉。
- read path：不到 15 个字的 prompt 不注入。更长的 prompt 一律注入 5 条，哪怕和之前的工作无关。中文问法要匹配英文记录时，BM25 那一路帮不上忙（「去重」对 deduplication）。
- docs lookup：到达服务的 prompt 一律注入 3 节，哪怕和文档无关。新 repo 的第一句 prompt 拿不到文档。冷启动时第一句 prompt 可能超过 hook 的 12 秒上限，这一句什么都不注入。

## License

[MIT](LICENSE)
