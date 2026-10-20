import pytest

import core
from eval import QUESTIONS, main, metrics, rank_of, report

OBS = {"db": "data/exp3-obs.db", "unit": "observation",
       "embed": {"model": "BAAI/bge-m3"}, "search": {"mode": "hybrid"}}
TURNS = {**OBS, "db": "data/exp3-turns.db", "unit": "turn"}


def test_rank_of_takes_the_best_ranked_target():
    assert rank_of([5], [9, 5, 1]) == 2
    assert rank_of([1, 5], [9, 5, 1]) == 2
    assert rank_of([5, 7], [1, 2]) is None


def test_metrics():
    hits, mrr = metrics([1, 2, None, 10, None, None, None, None, None, 5])
    assert hits == 3
    assert abs(mrr - (1 + 1 / 2 + 1 / 10 + 1 / 5) / 10) < 1e-9


def test_report_dense_mode_matches_the_baseline_format():
    rows = [(a, q, None, None, {"dense": None}) for a, _, _, q in QUESTIONS]
    rows[2] = (29107, QUESTIONS[2][3], 7, (29100, "Some Other Title"), {"dense": 7})
    expected = "\n".join([
        "corpus  15479 条 observation（id ≤ 37233）",
        "dense   bge-m3",
        "bm25    关",
        "",
        " #  answer  名次  question",
        " 1  25016 >100   列表页翻页的 D1 race condition 是怎么挡的",
        " 2  22998 >100   C 引号路径导致「从未存在过」误判那个 bug",
        " 3  29107    7   attention 为什么 CPU 用 SDPA、GPU 用 flash-attn varlen、不留 CUDA fallback",
        "                 ↳ 第 1 是 29100 Some Other Title",
        " 4  24111 >100   kill_stale_server 为什么要先核 PID 对应的进程名",
        " 5  23180 >100   上线的 runbook 里，为什么要先把数据灌进去再发版",
        " 6  23764 >100   给我讲代码的时候，解释是跟在代码后面写，还是另起一行写",
        " 7  29948 >100   架构图 validation 没过的那几轮，最后怎么改才通过的",
        " 8  26884 >100   那一大堆改动拆成 7 个 PR，按什么顺序排",
        " 9  29968 >100   部署流程里数据库配置缺了的时候，为什么要让它一开头就报错",
        "10  35863 >100   DeepSeek 返回 400 那次，trace 里连着好几条 assistant 消息，后来是怎么处理的",
        "",
        "hit@5  0/10    MRR  0.014",
    ])
    assert report({**OBS, "search": {"mode": "dense"}}, (15479, 37233), rows) == expected


def test_report_observation_side_matches_sample():
    rows = [(a, q, None, None, {"dense": None, "bm25": None}) for a, _, _, q in QUESTIONS]
    rows[0] = (25016, QUESTIONS[0][3], 3, None, {"dense": 2, "bm25": 7})
    rows[4] = (23180, QUESTIONS[4][3], 12, (31000, "Some Other Title"), {"dense": 9, "bm25": None})
    expected = "\n".join([
        "corpus  15479 条 observation（id ≤ 37233）",
        "dense   bge-m3",
        "bm25    开 · RRF k=60",
        "",
        " #  answer  名次  dense  bm25  question",
        " 1  25016    3        2     7  列表页翻页的 D1 race condition 是怎么挡的",
        " 2  22998 >100     >100  >100  C 引号路径导致「从未存在过」误判那个 bug",
        " 3  29107 >100     >100  >100  attention 为什么 CPU 用 SDPA、GPU 用 flash-attn varlen、不留 CUDA fallback",
        " 4  24111 >100     >100  >100  kill_stale_server 为什么要先核 PID 对应的进程名",
        " 5  23180   12        9  >100  上线的 runbook 里，为什么要先把数据灌进去再发版",
        "                               ↳ 第 1 是 31000 Some Other Title",
        " 6  23764 >100     >100  >100  给我讲代码的时候，解释是跟在代码后面写，还是另起一行写",
        " 7  29948 >100     >100  >100  架构图 validation 没过的那几轮，最后怎么改才通过的",
        " 8  26884 >100     >100  >100  那一大堆改动拆成 7 个 PR，按什么顺序排",
        " 9  29968 >100     >100  >100  部署流程里数据库配置缺了的时候，为什么要让它一开头就报错",
        "10  35863 >100     >100  >100  DeepSeek 返回 400 那次，trace 里连着好几条 assistant 消息，后来是怎么处理的",
        "",
        "hit@5  1/10    MRR  0.042",
        "只 dense  hit@5 1/10    MRR 0.061",
        "只 bm25   hit@5 0/10    MRR 0.014",
    ])
    assert report(OBS, (15479, 37233), rows) == expected


def test_report_turn_side_matches_sample():
    rows = [(a, q, None, None, {"dense": None, "bm25": None}) for a, _, _, q in QUESTIONS]
    rows[1] = (22998, QUESTIONS[1][3], None, (1001234, "Try again"), {"dense": 50, "bm25": None})
    rows[9] = (35863, QUESTIONS[9][3], 1, None, {"dense": 1, "bm25": 4})
    expected = "\n".join([
        "corpus  3695 个 turn（id ≤ 1003695）",
        "dense   bge-m3",
        "bm25    开 · RRF k=60",
        "",
        " #  answer  名次  dense  bm25  question",
        " 1  25016 >100     >100  >100  列表页翻页的 D1 race condition 是怎么挡的",
        " 2  22998 >100       50  >100  C 引号路径导致「从未存在过」误判那个 bug",
        "                               ↳ 第 1 是 1001234 Try again",
        " 3  29107 >100     >100  >100  attention 为什么 CPU 用 SDPA、GPU 用 flash-attn varlen、不留 CUDA fallback",
        " 4  24111 >100     >100  >100  kill_stale_server 为什么要先核 PID 对应的进程名",
        " 5  23180 >100     >100  >100  上线的 runbook 里，为什么要先把数据灌进去再发版",
        " 6  23764 >100     >100  >100  给我讲代码的时候，解释是跟在代码后面写，还是另起一行写",
        " 7  29948 >100     >100  >100  架构图 validation 没过的那几轮，最后怎么改才通过的",
        " 8  26884 >100     >100  >100  那一大堆改动拆成 7 个 PR，按什么顺序排",
        " 9  29968 >100     >100  >100  部署流程里数据库配置缺了的时候，为什么要让它一开头就报错",
        "10  35863    1        1     4  DeepSeek 返回 400 那次，trace 里连着好几条 assistant 消息，后来是怎么处理的",
        "",
        "hit@5  1/10    MRR  0.100",
        "只 dense  hit@5 1/10    MRR 0.102",
        "只 bm25   hit@5 1/10    MRR 0.025",
    ])
    assert report(TURNS, (3695, 1003695), rows) == expected


IDS = [24974, 1000642, 25016]  # 第 1 题 turn 里的另一条 observation、第 1 题的答案 turn、第 1 题的答案


class FakeMemory:
    """每题都搜出 IDS 这三条"""

    def __init__(self, cfg):
        self.store = self

    def stats(self):
        return (3, 1000642)

    def search(self, question, k):
        return [{"id": i, "title": f"标题 {i}"} for i in IDS]

    def rankings(self, question, k):
        return {"dense": IDS, "bm25": IDS}


@pytest.mark.parametrize(("unit", "rank"), [("observation", 1), ("turn", 2)])
def test_main_judges_each_side_by_the_answer_turn(tmp_path, monkeypatch, capsys, unit, rank):
    monkeypatch.setattr(core, "Memory", FakeMemory)
    path = tmp_path / "c.toml"
    path.write_text(f'db = "x.db"\nunit = "{unit}"\n\n[embed]\nmodel = "BAAI/bge-m3"\n\n'
                    '[search]\nmode = "hybrid"\n')
    main(str(path))
    assert capsys.readouterr().out.splitlines()[5] == (
        f" 1  25016{rank:>5}{rank:>9}{rank:>6}  列表页翻页的 D1 race condition 是怎么挡的")


def test_report_prints_the_rerank_line_when_configured():
    rows = [(a, q, None, None, {"dense": None, "bm25": None}) for a, _, _, q in QUESTIONS]
    cfg = {**OBS, "db": "data/exp4-dedup.db", "search": {"mode": "hybrid", "rerank": "BAAI/bge-reranker-v2-m3"}}
    lines = report(cfg, (15222, 37233), rows).splitlines()
    assert lines[:4] == ["corpus  15222 条 observation（id ≤ 37233）", "dense   bge-m3",
                         "bm25    开 · RRF k=60", "rerank  bge-reranker-v2-m3 · 前 20 条"]
    assert lines[4] == ""
