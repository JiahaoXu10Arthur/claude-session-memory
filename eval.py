import sys
import tomllib

import core

# 实验 3 起的十题，spec 第七节「实验 3 的样张」：答案 observation、答案 turn、这个 turn 压出来的 observation、question。
# B 边认答案 turn；A 边认这个 turn 压出来的任何一条，名次按排得最靠前的那条算
QUESTIONS = [
    (25016, 1000642, [
        24974, 24977, 24978, 24979, 24982, 24986, 24987, 24990, 24991, 24992, 24994, 24995,
        24996, 24997, 24998, 25000, 25001, 25002, 25003, 25004, 25005, 25006, 25007, 25008,
        25009, 25010, 25011, 25012, 25014, 25015, 25016, 25017, 25018, 25019,
    ], "列表页翻页的 D1 race condition 是怎么挡的"),
    (22998, 1000357, [
        22971, 22972, 22973, 22974, 22975, 22976, 22977, 22978, 22979, 22980, 22981, 22982,
        22983, 22984, 22985, 22986, 22988, 22989, 22992, 22998, 22999, 23004, 23005, 23006,
        23007, 23008, 23009, 23010, 23011, 23012, 23013, 23014, 23017, 23019, 23020, 23021,
        23022, 23023, 23024, 23026, 23028, 23035, 23036, 23037, 23038, 23040, 23041, 23042,
        23043, 23044, 23045, 23047, 23048, 23049, 23051, 23054, 23056, 23058, 23062, 23065,
        23066, 23067, 23069, 23070, 23071, 23074, 23076, 23077, 23078, 23079, 23080, 23081,
        23082, 23083, 23084, 23085, 23086, 23087, 23088, 23089, 23090, 23091, 23092,
    ], "C 引号路径导致「从未存在过」误判那个 bug"),
    (29107, 1001998, [
        29103, 29104, 29106, 29107, 29108, 29109, 29110, 29111, 29112, 29113, 29114, 29115,
    ], "attention 为什么 CPU 用 SDPA、GPU 用 flash-attn varlen、不留 CUDA fallback"),
    (24111, 1000540, [
        24071, 24072, 24073, 24074, 24075, 24076, 24077, 24078, 24079, 24080, 24081, 24082,
        24083, 24084, 24085, 24086, 24087, 24088, 24089, 24090, 24091, 24092, 24093, 24094,
        24095, 24096, 24097, 24098, 24099, 24100, 24101, 24102, 24103, 24104, 24105, 24106,
        24107, 24108, 24109, 24110, 24111, 24112, 24113, 24114, 24115, 24116, 24117, 24118,
        24119, 24120, 24121, 24122, 24123, 24124, 24125, 24126, 24127, 24128, 24129,
    ], "kill_stale_server 为什么要先核 PID 对应的进程名"),
    (23180, 1000380, [
        23178, 23180,
    ], "上线的 runbook 里，为什么要先把数据灌进去再发版"),
    (23764, 1000494, [
        23764,
    ], "给我讲代码的时候，解释是跟在代码后面写，还是另起一行写"),
    (29948, 1002308, [
        29945, 29946, 29947, 29948, 29949, 29950, 29952, 29953, 29954, 29955, 29956,
    ], "架构图 validation 没过的那几轮，最后怎么改才通过的"),
    (26884, 1001062, [
        26875, 26876, 26877, 26878, 26879, 26880, 26881, 26882, 26883, 26884,
    ], "那一大堆改动拆成 7 个 PR，按什么顺序排"),
    (29968, 1002307, [
        29951, 29957, 29958, 29959, 29960, 29961, 29962, 29963, 29964, 29965, 29966, 29967,
        29968,
    ], "部署流程里数据库配置缺了的时候，为什么要让它一开头就报错"),
    (35863, 1003020, [
        35850, 35851, 35852, 35853, 35854, 35855, 35856, 35857, 35858, 35859, 35860, 35861,
        35862, 35863, 35864, 35865, 35866, 35867, 35868, 35869, 35870, 35871, 35872,
    ], "DeepSeek 返回 400 那次，trace 里连着好几条 assistant 消息，后来是怎么处理的"),
]

DEPTH = 100


def rank_of(targets: list[int], ids: list[int]) -> int | None:
    """targets 里排得最靠前的那个的名次"""
    ranks = [ids.index(t) + 1 for t in targets if t in ids]
    return min(ranks) if ranks else None


def metrics(ranks: list[int | None]) -> tuple[int, float]:
    hits = sum(1 for r in ranks if r is not None and r <= 5)
    mrr = sum(1 / r if r is not None else 0 for r in ranks) / len(ranks)
    return hits, mrr


def field(rank: int | None) -> str:
    return str(rank) if rank is not None else ">100"


def report(
    cfg: dict,
    stats: tuple[int, int],
    rows: list[tuple[int, str, int | None, tuple[int, str] | None, dict[str, int | None]]],
) -> str:
    count, max_id = stats
    hybrid = cfg["search"]["mode"] == "hybrid"
    unit = "个 turn" if cfg["unit"] == "turn" else "条 observation"
    lines = [
        f"corpus  {count} {unit}（id ≤ {max_id}）",
        f"dense   {cfg['embed']['model'].split('/')[-1]}",
        f"bm25    开 · RRF k={core.RRF_K}" if hybrid else "bm25    关",
        *([f"rerank  {cfg['search']['rerank'].split('/')[-1]} · 前 {core.RERANK_TOP} 条"]
          if "rerank" in cfg["search"] else []),
        "",
        " #  answer  名次  dense  bm25  question" if hybrid else " #  answer  名次  question",
    ]
    for i, (answer, question, rank, top1, ranks) in enumerate(rows, 1):
        if hybrid:
            lines.append(f"{i:>2}  {answer:>5}{field(rank):>5}{field(ranks['dense']):>9}"
                         f"{field(ranks['bm25']):>6}  {question}")
        else:
            lines.append(f"{i:>2}  {answer:>5}{field(rank):>5}   {question}")
        if top1 is not None:
            top1_id, top1_title = top1
            lines.append(" " * (31 if hybrid else 17) + f"↳ 第 1 是 {top1_id} {top1_title}")
    lines.append("")
    hits, mrr = metrics([rank for _, _, rank, _, _ in rows])
    lines.append(f"hit@5  {hits}/{len(rows)}    MRR  {mrr:.3f}")
    if hybrid:
        for name, label in [("dense", "只 dense  "), ("bm25", "只 bm25   ")]:
            hits, mrr = metrics([ranks[name] for *_, ranks in rows])
            lines.append(f"{label}hit@5 {hits}/{len(rows)}    MRR {mrr:.3f}")
    return "\n".join(lines)


def main(config_path: str) -> None:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    mem = core.Memory(cfg)
    rows = []
    for answer, turn, turn_obs, question in QUESTIONS:
        targets = [turn] if cfg["unit"] == "turn" else turn_obs
        hits = mem.search(question, DEPTH)
        rank = rank_of(targets, [h["id"] for h in hits])
        top1 = (hits[0]["id"], hits[0]["title"]) if (rank is None or rank > 5) and hits else None
        ranks = {name: rank_of(targets, ids) for name, ids in mem.rankings(question, DEPTH).items()}
        rows.append((answer, question, rank, top1, ranks))
    print(report(cfg, mem.store.stats(), rows))


if __name__ == "__main__":
    main(sys.argv[1])
