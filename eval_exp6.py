"""实验 6 的 eval：同一批题在 A 边（exp6-theirs.db）和 B 边（exp6-ours.db）各跑一遍，按 turn 判，再按手标的答案记录判。
    uv run python eval_exp6.py"""
import tomllib

import core
import eval as ev

# (question, session 前缀, 答案 turn 的 start, 手标的答案记录 {"A": [...], "B": [...]}；压完读过两边再填，空着就只按 turn 判)
QUESTIONS: list[tuple[str, str, int, dict[str, list[int]]]] = [
    # 十题（9/29 挑的）。答案 turn 是这事第一次发生的那轮，见 sdd/2026-09-27-exp6/answer-turns.md。
    # 手标：压完读两边答案 turn 的记录后标真正回答这题的那几条；空着就只按 turn 判
    ("turn 的边界为什么不能按下一条用户消息切", "81acef29", 2660981, {"A": [37798, 37799], "B": [37845, 37846, 37849]}),  # 原因在前一轮 2146260 查出，决定在这一轮；两轮的记录都算
    ("一个 turn 怎么切块进向量", "81acef29", 4207147, {"A": [37860], "B": [37866]}),
    ("去重是按什么规矩合并的，删了多少", "f1d28823", 860754, {"A": [38263], "B": [38202]}),
    ("rerank 输入为什么截到 1500 字", "f1d28823", 5604893, {"A": [38391, 38395, 38405], "B": [38259]}),  # 38277、38394 只说做了什么没说为什么，核查后去掉
    ("Ollama 的 /v1 接口怎么关 thinking", "f1d28823", 5604893, {"A": [38388], "B": [38260, 38381]}),  # A 的 38380 同一轮，说的是反的（关不掉），不算
    ("subagent 的 transcript 归到哪个 turn，怎么定的", "aeb3f3c6", 83395, {"A": [38512, 38513], "B": [37235, 37255, 37403]}),  # 核查后补 A：38512/38513 写了 toolUseId attribution
    ("Ollama 挂了的时候，压到一半的 turn 怎么处理", "aeb3f3c6", 8477445, {"A": [38631, 38641, 38644, 38646], "B": [37324, 37330, 37417, 37448, 37454]}),  # 核查后补 A：38631 Multi-Segment Skip、38646 的 facts 写了 outage 后段丢
    ("压缩 prompt 为什么要求只写英文", "aeb3f3c6", 5639111, {"A": [38575, 38579], "B": [37298, 37304, 37436]}),
    ("docs/ 是怎么从 git 历史里抹掉的，备份呢", "aeb3f3c6", 5304736, {"A": [38546, 38547], "B": [37279, 37280]}),
    ("serve.py 闲多久自己退，为什么", "aeb3f3c6", 11872717, {"A": [38731, 38732], "B": [37360]}),
]

SIDES = [("A", "configs/exp6-theirs.toml", "claude-mem"), ("B", "configs/exp6-ours.toml", "写入半")]


def turn_obs(store, session: str, start: int) -> list[int]:
    row = store.conn.execute("SELECT id FROM turns WHERE session_id LIKE ? AND start = ?", (session + "%", start)).fetchone()
    if row is None:
        return []
    return [r[0] for r in store.conn.execute("SELECT id FROM obs WHERE turn_id = ?", (row[0],))]


def main() -> None:
    mem = None
    for side, path, label in SIDES:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        if mem is None:
            mem = core.Memory(cfg)
        else:
            import store
            mem.store = store.Store(cfg["db"], mem.embedder.dim)
        count, _ = mem.store.stats()
        print(f"\n=== {side} 边 {label}  corpus {count} 条")
        print(" #  turn  手标  dense  bm25  答案 turn 条数  question")
        by_turn, by_hand = [], []
        for i, (q, session, start, hand) in enumerate(QUESTIONS, 1):
            targets = turn_obs(mem.store, session, start)
            hits = mem.search(q, ev.DEPTH)
            ids = [h["id"] for h in hits]
            r_turn = ev.rank_of(targets, ids)
            r_hand = ev.rank_of(hand.get(side, []), ids) if hand.get(side) else None
            ranks = {n: ev.rank_of(targets, l) for n, l in mem.rankings(q, ev.DEPTH).items()}
            by_turn.append(r_turn); by_hand.append(r_hand)
            print(f"{i:>2}{ev.field(r_turn):>6}{ev.field(r_hand) if hand.get(side) else '·':>6}{ev.field(ranks['dense']):>7}"
                  f"{ev.field(ranks['bm25']):>6}{len(targets):>8}        {q}")
            h = hits[0]
            tag = "" if h["id"] in targets else "（不是答案 turn）"
            print(f"      ↳ 第 1 是 {h['id']}{tag}: {h['title']} — {h['subtitle']}")
        hits5, mrr = ev.metrics(by_turn)
        print(f"按 turn 判  hit@5 {hits5}/{len(by_turn)}  MRR {mrr:.3f}")
        if any(by_hand):
            hits5, mrr = ev.metrics(by_hand)
            print(f"按手标判    hit@5 {hits5}/{len(by_hand)}  MRR {mrr:.3f}")


if __name__ == "__main__":
    main()
