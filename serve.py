"""常驻服务：加载一次模型，hook 只发请求。启动顺序：先接日志，再 bind 端口（bind 不上就退出，端口就是互斥锁），最后才加载模型。
写入半：/write 只入队，一个 worker 顺序压，最新的先压；每 10 分钟扫一遍最近改过的 transcript 放到队列底部"""
import collections
import json
import os
import socketserver
import sys
import threading
import time
import tomllib
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import docs
import transcript

NAME = "session-mem"
DEPTH = 100
TOP = 5
CUT = "2026-09-24T15:56:30.162Z"  # 初始库最后一条 observation 的时间：这之前的 turn 已经在库里，不再压
LAUNCH = "2026-09-26T23:30:00.000Z"  # 写入半上线的时刻：扫尾发现的、这之前开始的 session 不回灌
SWEEP_EVERY = 600
IDLE_EXIT = 1800  # 这么久没收到请求、worker 也闲着，就退出：用户不在这个项目干活时不占 4.6 GB 内存
DOC_TOP = 3  # 第一面：一句 prompt 最多注这么多节文档
INJECT_CAP = 10_000  # Claude Code 每条注入的上限
MAX_FAILURES = 3  # 同一个单位连续出错这么多次：压 turn 出错的跳过这个 turn；别的错就不再放回队列，等下次扫尾


def main(config_path: str) -> None:
    os.chdir(Path(__file__).resolve().parent)  # hook 拉起服务时 cwd 是那个 session 的 repo；config 里的相对路径按本 repo 算
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    log = open(Path(cfg["db"]).with_name("serve.log"), "a", buffering=1)
    sys.stdout = sys.stderr = log
    socketserver.TCPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", cfg["serve"]["port"]), Handler)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 模型都在本机 cache 里，不去问 HF
    import core  # bind 之后才 import：torch 要 3 秒、模型要十几秒，这期间端口已经占住
    Handler.mem = core.Memory(cfg)
    Handler.injected = {}
    Handler.worker = Worker(Handler.mem, cfg.get("docs", {}).get("dir"))
    threading.Thread(target=Handler.worker.run, daemon=True).start()
    if sweep_enabled(cfg):  # 扫尾会拉起 Ollama 占 6.7 GB，只在做写入半的实验时开
        threading.Thread(target=Handler.worker.sweep_forever, daemon=True).start()
    threading.Thread(target=exit_when_idle, args=(Handler,), daemon=True).start()
    print("ready", flush=True)
    server.serve_forever()


def sweep_enabled(cfg: dict) -> bool:
    return bool(cfg.get("write", {}).get("sweep", False))


def exit_when_idle(handler) -> None:
    while True:
        time.sleep(60)
        if time.time() - handler.last_request > IDLE_EXIT and handler.worker.idle():
            print("idle, exit", flush=True)
            os._exit(0)


class Worker:
    """一个队列、一个线程。hook 的请求放队头，扫尾放队尾；一次只压一个单位（一个 turn、一段续篇、一份 subagent）"""

    def __init__(self, mem, docs_dir: str | None = None):
        self.mem = mem
        self.docs_dir = docs_dir
        self.stores = {}  # repo 根 → 那个 repo 的文档库，懒加载
        self.queue = collections.deque()
        self.ready = threading.Condition()
        self.failures = collections.Counter()
        self.busy = False

    def idle(self) -> bool:
        with self.ready:
            return not self.queue and not self.busy

    def push(self, path: str, front: bool) -> None:
        with self.ready:
            if path not in self.queue:
                (self.queue.appendleft if front else self.queue.append)(path)
                self.ready.notify()

    def run(self) -> None:
        while True:
            self.once()

    def once(self) -> None:
        """等到队列有东西，取队头压一个单位；出错记一次，连续 MAX_FAILURES 次就不放回、计数清零"""
        with self.ready:
            while not self.queue:
                self.ready.wait()
            path = self.queue.popleft()
            self.busy = True
        try:
            again = self.step(path)
            self.failures.pop(path, None)
        except Exception:
            traceback.print_exc()
            self.failures[path] += 1
            again = self.failures[path] < MAX_FAILURES
            if not again:
                del self.failures[path]
        finally:
            with self.ready:
                self.busy = False
        if again:
            self.push(path, front=False)

    def sweep_forever(self) -> None:
        while True:
            time.sleep(SWEEP_EVERY)
            try:
                for path in transcript.sweep():
                    self.push(path, front=False)
            except Exception:  # 比如 glob 之后文件被删了；线程不能死
                traceback.print_exc()

    def docs_store(self, root: str):
        with self.mem.lock:
            if root not in self.stores:
                import store
                path = os.path.join(self.docs_dir, os.path.basename(root) + ".db")
                if not os.path.exists(path):
                    print(f"docs {os.path.basename(root)}: 还没有库，这一句没有 docs 段，排队建", flush=True)
                self.stores[root] = store.Store(path, self.mem.embedder.dim)
            return self.stores[root]

    def index(self, root: str) -> None:
        """对比 mtime：变了的、新的文件重切重 embed，不在了的删；一个文件一个事务，文件之间放锁"""
        store = self.docs_store(root)
        known = store.cursors()
        current = docs.mtimes(root)
        for path, mtime in current.items():
            if known.get(path) != mtime:
                rel = os.path.relpath(path, root)
                self.mem.index(store, f"{rel} § ", docs.sections(path, root))
                store.set_cursor(path, mtime)
        for path in known.keys() - current.keys():
            self.mem.index(store, f"{os.path.relpath(path, root)} § ", [])
            store.del_cursor(path)
        print(f"docs {os.path.basename(root)}: {store.stats()[0]} 节", flush=True)

    def step(self, path: str) -> bool:
        """压这份文件的下一个单位；返回还要不要再来一次。docs: 开头的是索引一个 repo 的文档"""
        if path.startswith("docs:"):
            self.index(path[len("docs:"):])
            return False
        store = self.mem.store
        cursor = store.cursor(path)
        if cursor is None:  # 只有扫尾发现的文件会没游标（hook 的请求进来时就设了 0）
            started = transcript.started_at(path)
            if started is None or started < LAUNCH:  # 上线前就有的 session（或 fork 出来的空壳）：不回灌
                store.set_cursor(path, os.path.getsize(path))
                return False
            cursor = 0
        r = (transcript.next_subagent if "/subagents/" in path else transcript.next_turn)(path, cursor)
        if isinstance(r, int):
            if r != cursor:
                store.set_cursor(path, r)
            return False
        if r.created_at <= CUT:
            store.set_cursor(path, r.end)
            return True
        segments = transcript.windows(r)
        if len(r.text) > transcript.WINDOW * transcript.MAX_WINDOWS:
            print(f"write {r.session_id[:8]} start={r.start}: {len(r.text)} 字符，只压前 {len(segments)} 段，尾巴丢了", flush=True)
        meta = {"session_id": r.session_id, "start": r.start, "project": r.project,
                "created_at": r.created_at, "prompt": r.prompt}
        i = 0
        try:
            for i, seg in enumerate(segments, 1):
                t0 = time.time()
                out = self.mem.write(seg, {**meta, "cursor": (path, r.end) if i == len(segments) else None})
                print(f"write {r.session_id[:8]} start={r.start} 段 {i}/{len(segments)} → {len(out)} 条，{time.time() - t0:.0f} 秒", flush=True)
        except Exception:
            if i == 1 or self.failures[path] + 1 < MAX_FAILURES:  # 第 1 段就出错（多半是模型连不上）：放弃，等扫尾再来
                raise
            traceback.print_exc()  # 第 2 段起连续 MAX_FAILURES 次出错：这个 turn 跳过，不然前面几段每次重试都多写一遍
            print(f"write {r.session_id[:8]} start={r.start} 段 {i} 连续出错 {MAX_FAILURES} 次，跳过这个 turn", flush=True)
            store.set_cursor(path, r.end)
        return True


class Handler(BaseHTTPRequestHandler):
    mem = None
    worker = None
    last_request = time.time()
    injected = {}  # session_id → 这个 session 已经注过的文档节的 title

    def log_message(self, fmt, *args):
        print(self.address_string(), fmt % args, flush=True)

    def reply(self, code: int, body: str) -> None:
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200 if self.path == "/health" else 404, NAME if self.path == "/health" else "")

    def do_POST(self):
        Handler.last_request = time.time()
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/write":
            path = req["transcript_path"]
            with self.mem.lock:
                if self.mem.store.cursor(path) is None:
                    self.mem.store.set_cursor(path, 0)  # hook 来的 session 从头压（CUT 之前的 turn 会跳过）
            self.worker.push(path, front=True)
            self.reply(200, "")
            return
        if self.path != "/search":
            self.reply(404, "")
            return
        import core
        hits = self.mem.search(req["query"], DEPTH)
        hits = [h for h in hits if h["session_id"] != req["session_id"]][:TOP]
        text = core.format(hits) if hits else ""
        injected = []
        try:  # docs 那步出什么错都不能把 memory 段一起丢了
            root = docs.repo_root(req["cwd"]) if self.worker.docs_dir and req.get("cwd") else None
            if root:
                store = self.worker.docs_store(root)
                if docs.stale(root, store.cursors()):
                    self.worker.push("docs:" + root, front=False)
                seen = Handler.injected.setdefault(req["session_id"], set())
                dhits = [h for h in self.mem.search(req["query"], DEPTH, store=store, rerank=False)
                         if h["title"] not in seen][:DOC_TOP]
                while dhits:  # 两段合起来不能超 INJECT_CAP，超了从 docs 段末尾去整节
                    block = core.format_docs(dhits)
                    if len(text) + 2 + len(block) <= INJECT_CAP:
                        injected = [h["title"] for h in dhits]
                        text = f"{text}\n\n{block}" if text else block
                        break
                    dhits.pop()
        except Exception:
            traceback.print_exc()
        try:
            self.reply(200, text)
        except OSError:  # hook 等不及先关了连接（冷启动那一句常这样）：没送到，不算注过
            return
        if injected:
            Handler.injected[req["session_id"]].update(injected)


if __name__ == "__main__":
    main(sys.argv[1])
