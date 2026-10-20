import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import core
import embed
import serve
from tests.test_core import FakeEmbedder, FakeReranker, obs

ROOT = Path(__file__).parent.parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def config(tmp_path, port):
    path = tmp_path / "live.toml"
    path.write_text(f'db = "{tmp_path / "live.db"}"\nunit = "observation"\n\n[embed]\nmodel = "fake"\n\n'
                    f'[search]\nmode = "dense"\nrerank = "fake-reranker"\n\n[serve]\nport = {port}\n')
    return str(path)


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(embed, "Reranker", FakeReranker)
    port = free_port()
    path = config(tmp_path, port)
    with open(path, "rb") as f:
        import tomllib
        cfg = tomllib.load(f)
    m = core.Memory(cfg)
    m.add([obs(1, session_id="other", title="Gap rule", narrative="stint gap"),
           obs(2, session_id="mine", title="Mine", narrative="stint gap too")])
    monkeypatch.setattr(sys, "stdout", sys.stdout)  # serve.main 会把 stdout 换成日志文件，测试完还原
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    t = threading.Thread(target=serve.main, args=(path,), daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if urllib.request.urlopen(f"{base}/health", timeout=1).read() == b"session-mem":
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        import time
        time.sleep(0.05)
    else:
        pytest.fail("server did not come up")
    return base, path


def post(base, query, session_id):
    body = json.dumps({"query": query, "session_id": session_id}).encode()
    req = urllib.request.Request(f"{base}/search", body, {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5).read().decode()


def test_health_says_who_it_is(server):
    base, _ = server
    assert urllib.request.urlopen(f"{base}/health").read() == b"session-mem"


def test_search_excludes_the_asking_session_and_formats(server):
    base, _ = server
    out = post(base, "stint gap", "mine")
    assert out.splitlines()[0] == "memory：和这句话相关的 1 条"
    assert "Gap rule" in out and "Mine" not in out


def test_search_asked_from_the_other_session_flips_the_result(server):
    base, _ = server
    out = post(base, "stint gap", "other")
    assert "Mine" in out and "Gap rule" not in out


def run_hook(path, prompt, session_id="s"):
    return subprocess.run([sys.executable, str(ROOT / "hooks" / "prompt.py"), path],
                          input=json.dumps({"prompt": prompt, "session_id": session_id}),
                          capture_output=True, text=True, timeout=30)


def test_prompt_hook_prints_the_hits(server):
    base, path = server
    r = run_hook(path, "how was the stint gap handled last time", "mine")
    assert r.returncode == 0 and r.stdout.startswith("memory：和这句话相关的 1 条")


def test_prompt_hook_skips_short_prompts(server):
    base, path = server
    r = run_hook(path, "你定", "mine")
    assert r.returncode == 0 and r.stdout == ""


def test_prompt_hook_ignores_notifications_and_does_not_start_a_server(tmp_path, monkeypatch):
    # 后台任务的通知、system-reminder 也走 UserPromptSubmit；它们不是用户在问，不查也不拉服务
    path = config(tmp_path, free_port())
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a[0]))
    import importlib.util
    spec = importlib.util.spec_from_file_location("prompt", ROOT / "hooks" / "prompt.py")
    prompt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt)
    for text in ("<task-notification>\n<task-id>abc</task-id> agent finished with a long report", "<system-reminder>\nsomething long enough"):
        monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps({"prompt": text, "session_id": "s"})))
        prompt.main(path)
    assert started == []


def test_prompt_hook_stays_quiet_when_the_search_times_out(tmp_path, monkeypatch):
    # 服务忙（worker 在 embed、GPU 被压缩占着）时 /search 超过 12 秒：不注入也不报错，不能把 traceback 甩给用户
    path = config(tmp_path, free_port())
    import importlib.util
    spec = importlib.util.spec_from_file_location("prompt", ROOT / "hooks" / "prompt.py")
    prompt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt)
    calls = []

    def urlopen(req, timeout=None):
        calls.append(req if isinstance(req, str) else req.full_url)
        if len(calls) == 1:
            return __import__("io").BytesIO(b"session-mem")  # /health
        raise TimeoutError("timed out")  # /search

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps({"prompt": "a question long enough", "session_id": "s"})))
    prompt.main(path)  # 不抛
    assert len(calls) == 2


def test_prompt_hook_starts_the_server_when_nothing_answers(tmp_path, monkeypatch):
    path = config(tmp_path, free_port())
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a[0]))
    import importlib.util
    spec = importlib.util.spec_from_file_location("prompt", ROOT / "hooks" / "prompt.py")
    prompt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps({"prompt": "a question long enough", "session_id": "s"})))
    prompt.main(path)
    assert started and started[0][1].endswith("serve.py")


def test_prompt_hook_does_not_start_a_second_server_while_the_first_is_loading(tmp_path, monkeypatch):
    # 加载模型时端口已经 listen 但没人 accept：/health 超时，不是拒绝连接
    port = free_port()
    path = config(tmp_path, port)
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))
        s.listen(1)
        started = []
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a[0]))
        import importlib.util
        spec = importlib.util.spec_from_file_location("prompt", ROOT / "hooks" / "prompt.py")
        prompt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompt)
        monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps({"prompt": "a question long enough", "session_id": "s"})))
        prompt.main(path)
    assert started == []


def test_prompt_hook_only_starts_a_server_when_the_connection_is_refused(tmp_path, monkeypatch):
    # accept 队列满了以后 urlopen 抛的是 URLError(TimeoutError)，也算「在加载」
    path = config(tmp_path, free_port())
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a[0]))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError(TimeoutError())))
    import importlib.util
    spec = importlib.util.spec_from_file_location("prompt", ROOT / "hooks" / "prompt.py")
    prompt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps({"prompt": "a question long enough", "session_id": "s"})))
    prompt.main(path)
    assert started == []


def test_serve_loads_models_offline(server):
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "1"


# ---- 写入半：Worker、/write、hooks/stop.py

import os

import transcript
from tests.test_core import FakeLLM
from tests.test_transcript import SID, assistant, human, summary, t, write


def memory(tmp_path, monkeypatch, answers):
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(core.llm, "LLM", FakeLLM)
    m = core.Memory({"db": str(tmp_path / "m.db"), "embed": {"model": "fake"}, "search": {"mode": "dense"},
                     "llm": {"base_url": "http://x", "model": "m"}})
    m.llm.lock = m.lock
    m.llm.answers = list(answers)
    return m


def one_obs(title):
    return json.dumps({"observations": [{"title": title, "subtitle": "s", "narrative": "n", "facts": [], "files": [], "type": "change"}]})


def session_file(tmp_path, day="2026-09-25"):
    """一个 Stop 过的 turn 加一句还没 Stop 的 prompt；t() 的日期是 9/25（CUT 之后），要造 CUT 之前的换 day"""
    rows = [human(t("10:00:00"), "first prompt"), assistant(t("10:00:05"), "answer " * 20), summary(t("10:00:06")),
            human(t("10:01:00"), "second, not stopped yet")]
    return write(tmp_path / f"{SID}.jsonl", [{**r, "timestamp": day + r["timestamp"][10:]} for r in rows])


def test_step_compresses_one_turn_and_moves_the_cursor_only_after_the_last_window(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "WINDOW", 80)
    m = memory(tmp_path, monkeypatch, [one_obs("A"), one_obs("B")])
    path = session_file(tmp_path)
    m.store.set_cursor(path, 0)
    w = serve.Worker(m)
    assert w.step(path) is True  # 还要再来：文件里还有第二句
    assert [o["title"] for o in m.store.get([1, 2])] == ["A", "B"]
    turn = transcript.next_turn(path, 0)
    segments = transcript.windows(turn)
    assert [p[len(core.PROMPT):-len(core.PROMPT_TAIL)] for p, _ in m.llm.calls] == segments  # 两段，第 2 段带 prompt 头
    assert m.store.cursor(path) == turn.end
    assert m.store.conn.execute("SELECT text FROM turns").fetchone()[0] == "".join(segments)
    assert w.step(path) is False  # 第二句还没 Stop：游标不动
    assert m.store.cursor(path) == turn.end and len(m.llm.calls) == 2


def test_step_skips_turns_already_in_the_initial_db(tmp_path, monkeypatch):
    m = memory(tmp_path, monkeypatch, [])
    path = session_file(tmp_path, day="2026-09-20")  # CUT 之前
    m.store.set_cursor(path, 0)
    assert serve.Worker(m).step(path) is True
    assert m.store.cursor(path) == transcript.next_turn(path, 0).end and m.llm.calls == []


def test_step_without_cursor_skips_files_older_than_the_service_and_compresses_new_ones(tmp_path, monkeypatch):
    m = memory(tmp_path, monkeypatch, [one_obs("A")])
    path = session_file(tmp_path)
    w = serve.Worker(m)
    monkeypatch.setattr(serve, "LAUNCH", "2026-09-26T00:00:00.000Z")  # 上线比这个 session 晚：上线前的文件，游标设到末尾
    assert w.step(path) is False
    assert m.store.cursor(path) == os.path.getsize(path) and m.llm.calls == []
    fork = write(tmp_path / "fork.jsonl", [{"type": "custom-title", "sessionId": "fork"}])  # 自己的行没 timestamp
    assert w.step(fork) is False and m.store.cursor(fork) == os.path.getsize(fork)
    fresh = write(tmp_path / "fresh" / f"{SID}.jsonl", [])
    os.rename(path, fresh)
    monkeypatch.setattr(serve, "LAUNCH", "2026-09-25T00:00:00.000Z")  # 上线在前，session 后开：从 0 压
    assert w.step(fresh) is True
    assert len(m.llm.calls) == 1 and m.store.cursor(fresh) == transcript.next_turn(fresh, 0).end


def test_step_stores_the_offset_when_there_is_nothing_to_compress(tmp_path, monkeypatch):
    m = memory(tmp_path, monkeypatch, [])
    path = write(tmp_path / f"{SID}.jsonl", [{**human(t("10:00:00"), "claude -p"), "origin": {}}, summary(t("10:00:01"))])  # 有 summary 没起点
    m.store.set_cursor(path, 0)
    assert serve.Worker(m).step(path) is False
    assert m.store.cursor(path) == os.path.getsize(path) and m.llm.calls == []


def test_step_routes_subagent_files(tmp_path, monkeypatch):
    m = memory(tmp_path, monkeypatch, [one_obs("sub")])
    parent = session_file(tmp_path)
    sub = write(tmp_path / SID / "subagents" / "agent-a.jsonl",  # 父的起点之后开始、summary（10:00:06）之前写完
                [{**human(t("10:00:01"), "brief"), "isSidechain": True}, assistant(t("10:00:02"), "report")])
    monkeypatch.setattr(transcript, "SUBAGENT_IDLE", 0)
    m.store.set_cursor(sub, 0)
    assert serve.Worker(m).step(sub) is True
    assert m.store.get([1])[0]["title"] == "sub" and m.store.cursor(sub) == os.path.getsize(sub)
    assert m.store.conn.execute("SELECT start FROM turns").fetchone()[0] == transcript.next_turn(parent, 0).start


def test_push_puts_hook_requests_first_and_never_duplicates():
    w = serve.Worker(None)
    w.push("/sweep1", front=False)
    w.push("/hook", front=True)
    w.push("/sweep1", front=True)
    w.push("/sweep2", front=False)
    assert list(w.queue) == ["/hook", "/sweep1", "/sweep2"]


def test_once_requeues_a_failing_path_at_the_back_and_gives_up_after_three(monkeypatch):
    w = serve.Worker(None)

    def boom(path):
        raise RuntimeError("malformed line")

    monkeypatch.setattr(w, "step", boom)
    w.push("/a", front=True)
    w.push("/b", front=False)
    w.once()
    assert list(w.queue) == ["/b", "/a"]  # 出错的放回底部
    w.queue.clear()
    w.push("/a", front=True)
    w.once()
    w.once()
    assert list(w.queue) == [] and "/a" not in w.failures  # 第 3 次不放回，计数清零（下次扫尾捞回来再给 3 次）
    w.push("/a", front=True)
    monkeypatch.setattr(w, "step", lambda path: False)
    w.once()
    assert list(w.queue) == [] and "/a" not in w.failures  # 成功且没剩的不放回
    monkeypatch.setattr(w, "step", lambda path: True)
    w.push("/a", front=True)
    w.once()
    assert list(w.queue) == ["/a"]  # 还有剩的放回


def test_step_skips_a_turn_whose_window_keeps_failing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(transcript, "WINDOW", 80)
    m = memory(tmp_path, monkeypatch, [one_obs("A")])  # 第 2 段没有答案：FakeLLM 抛 IndexError，当 Ollama 出错
    path = session_file(tmp_path)
    m.store.set_cursor(path, 0)
    w = serve.Worker(m)
    w.push(path, front=True)
    w.once()
    w.once()
    assert m.store.cursor(path) == 0 and w.failures[path] == 2  # 前两次：整个 turn 放回重来
    m.llm.answers = [one_obs("A")]
    w.once()
    assert m.store.cursor(path) == transcript.next_turn(path, 0).end and path not in w.failures  # 第 3 次：跳过这个 turn
    assert "段 2 连续出错 3 次，跳过这个 turn" in capsys.readouterr().out
    assert list(w.queue) == [path]  # 文件还有剩，接着来


def test_step_never_skips_a_turn_whose_first_window_fails(tmp_path, monkeypatch):
    m = memory(tmp_path, monkeypatch, [])  # 一个答案都没有：模型连不上，第 1 段就抛
    path = session_file(tmp_path)
    m.store.set_cursor(path, 0)
    w = serve.Worker(m)
    w.push(path, front=True)
    for _ in range(3):
        w.once()
    assert m.store.cursor(path) == 0 and list(w.queue) == [] and path not in w.failures  # 放弃、不挪游标，等扫尾捞回来
    assert m.store.conn.execute("SELECT COUNT(*) FROM turns").fetchone() == (0,)


def test_write_endpoint_sets_cursor_zero_and_hands_the_path_to_the_worker(server, tmp_path, monkeypatch):
    base, cfg_path = server
    seen = []
    monkeypatch.setattr(serve.Handler.worker, "step", lambda path: seen.append((path, serve.Handler.mem.store.cursor(path))) and False)
    push = serve.Handler.worker.push
    fronts = []
    monkeypatch.setattr(serve.Handler.worker, "push", lambda path, front: fronts.append(front) or push(path, front))
    path = session_file(tmp_path)  # 上线前就开始的 session 也从 0 压：CUT 之后的 turn 都要
    req = urllib.request.Request(f"{base}/write", json.dumps({"transcript_path": path}).encode(), {"Content-Type": "application/json"})
    assert urllib.request.urlopen(req, timeout=5).status == 200
    import time
    for _ in range(100):
        if seen:
            break
        time.sleep(0.05)
    assert seen == [(path, 0)]  # worker 拿到它时游标已经是 0（不是 None，不会走「上线前的文件」那支）
    assert fronts == [True]  # hook 的请求放队头


def test_stop_hook_posts_the_transcript_path(server, tmp_path):
    base, cfg_path = server
    path = str(tmp_path / "s.jsonl")
    open(path, "w").close()
    import time
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "hooks" / "stop.py"), cfg_path],
                       input=json.dumps({"session_id": "x", "transcript_path": path}), capture_output=True, text=True, timeout=20)
    assert r.returncode == 0 and r.stdout == ""
    assert time.time() - t0 >= 3  # 等 SETTLE 秒让 summary 行先落地
    assert serve.Handler.mem.store.cursor(path) == 0  # 返回时服务已经收到


def test_stop_hook_is_silent_when_no_server_answers_or_hangs_up(tmp_path):
    cfg_path = config(tmp_path, free_port())
    r = subprocess.run([sys.executable, str(ROOT / "hooks" / "stop.py"), cfg_path],
                       input=json.dumps({"session_id": "x", "transcript_path": "/nope.jsonl"}), capture_output=True, text=True, timeout=20)
    assert r.returncode == 0 and r.stdout == "" and r.stderr == ""
    port = free_port()
    s = socket.socket()
    s.bind(("127.0.0.1", port))
    s.listen()
    threading.Thread(target=lambda: s.accept()[0].close(), daemon=True).start()  # 接了就挂断，不回话
    (tmp_path / "b").mkdir()
    r = subprocess.run([sys.executable, str(ROOT / "hooks" / "stop.py"), config(tmp_path / "b", port)],
                       input=json.dumps({"session_id": "x", "transcript_path": "/nope.jsonl"}), capture_output=True, text=True, timeout=20)
    assert r.returncode == 0 and r.stderr == ""


def test_worker_is_idle_only_when_queue_is_empty_and_nothing_is_running(monkeypatch):
    w = serve.Worker(None)
    assert w.idle()
    w.push("/a", front=True)
    assert not w.idle()
    seen = []
    monkeypatch.setattr(w, "step", lambda path: seen.append(w.idle()) or False)
    w.once()
    assert seen == [False] and w.idle()  # 压的时候不算闲，压完队列空了才算


def test_service_exits_after_idling(monkeypatch):
    class H:
        last_request = time.time() - serve.IDLE_EXIT - 1
        worker = serve.Worker(None)

    monkeypatch.setattr(serve.time, "sleep", lambda s: None)
    exits = []
    monkeypatch.setattr(serve.os, "_exit", lambda code: exits.append(code) or (_ for _ in ()).throw(SystemExit))
    with pytest.raises(SystemExit):
        serve.exit_when_idle(H)
    assert exits == [0]
    H.worker.push("/a", front=True)  # 闲够久了但 worker 还有活：不退，睡了两轮就掐掉
    calls = []

    def sleep(s):
        calls.append(s)
        if len(calls) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(serve.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        serve.exit_when_idle(H)
    assert exits == [0] and calls == [60, 60]


# ---- 第一面：repo 文档那一路

DOC_LIMIT = 10_000


def repo_with_docs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".claude" / "adr").mkdir(parents=True)
    (repo / ".claude" / "adr" / "0001-gap.md").write_text("# Gap rule\n\nthe stint gap rule lives here\n")
    (repo / ".claude" / "adr" / "0002-pid.md").write_text("# Pid rule\n\ncheck the pid first\n")
    return repo


def config_with_docs(tmp_path, port):
    path = tmp_path / "live.toml"
    path.write_text(f'db = "{tmp_path / "live.db"}"\nunit = "observation"\n\n[embed]\nmodel = "fake"\n\n'
                    f'[search]\nmode = "dense"\nrerank = "fake-reranker"\n\n[serve]\nport = {port}\n\n'
                    f'[docs]\ndir = "{tmp_path / "docs"}"\n')
    return str(path)


@pytest.fixture
def docs_server(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(embed, "Reranker", FakeReranker)
    port = free_port()
    path = config_with_docs(tmp_path, port)
    import tomllib
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    m = core.Memory(cfg)
    m.add([obs(1, session_id="other", title="Mem gap", narrative="stint gap")])
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    threading.Thread(target=serve.main, args=(path,), daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if urllib.request.urlopen(f"{base}/health", timeout=1).read() == b"session-mem":
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.05)
    else:
        raise RuntimeError("server did not start")
    yield base, tmp_path


def post_search(base, query, session_id, cwd=None):
    body = {"query": query, "session_id": session_id}
    if cwd is not None:
        body["cwd"] = cwd
    req = urllib.request.Request(f"{base}/search", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=10).read().decode()


def wait_until(pred, seconds=10):
    for _ in range(int(seconds / 0.05)):
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_search_without_docs_config_is_unchanged(server):
    base, _ = server
    out = post_search(base, "stint gap", "mine", cwd="/nowhere")
    assert out.startswith("memory：") and "docs：" not in out


def test_first_prompt_in_a_repo_gets_no_docs_but_queues_the_index_and_the_next_one_gets_it(docs_server):
    base, tmp_path = docs_server
    repo = repo_with_docs(tmp_path)
    out = post_search(base, "stint gap", "s1", cwd=str(repo))
    assert out.startswith("memory：") and "docs：" not in out
    assert wait_until(lambda: (tmp_path / "docs" / "repo.db").exists() and
                      "docs：" in post_search(base, "stint gap", "s2", cwd=str(repo)))
    out = post_search(base, "stint gap", "s3", cwd=str(repo))
    mem, docs_ = out.split("\n\n", 1)
    assert mem.startswith("memory：和这句话相关的 1 条")
    # dense 那路对库里每一节都有个分，所以两份 ADR 都进了前 3；对的那份排第 1
    assert docs_.startswith("docs：和这句话相关的 2 节\n- [.claude/adr/0001-gap.md § Gap rule]\n  # Gap rule the stint gap rule lives here")


def test_same_session_does_not_get_the_same_section_twice(docs_server):
    base, tmp_path = docs_server
    repo = repo_with_docs(tmp_path)
    post_search(base, "stint gap", "warm", cwd=str(repo))
    assert wait_until(lambda: "docs：" in post_search(base, "stint gap", "probe", cwd=str(repo)))
    first = post_search(base, "stint gap", "same", cwd=str(repo))
    second = post_search(base, "stint gap", "same", cwd=str(repo))
    assert "Gap rule" in first and "docs：" not in second and second.startswith("memory：")


def test_changed_and_deleted_docs_are_reindexed(docs_server):
    base, tmp_path = docs_server
    repo = repo_with_docs(tmp_path)
    post_search(base, "pid", "a", cwd=str(repo))
    assert wait_until(lambda: "Pid rule" in post_search(base, "pid", "b", cwd=str(repo)))
    (repo / ".claude" / "adr" / "0002-pid.md").unlink()
    gap = repo / ".claude" / "adr" / "0001-gap.md"
    gap.write_text("# Gap rule\n\nnow about nfl\n")
    os.utime(gap, (time.time() + 5, time.time() + 5))
    post_search(base, "pid", "c", cwd=str(repo))  # 看到 mtime 变了，排队
    assert wait_until(lambda: "Pid rule" not in post_search(base, "pid", "d", cwd=str(repo)))
    assert "now about nfl" in post_search(base, "nfl", "e", cwd=str(repo))


def test_response_is_capped_at_ten_thousand_chars_by_dropping_doc_sections(docs_server):
    base, tmp_path = docs_server
    repo = repo_with_docs(tmp_path)
    (repo / ".claude" / "adr" / "0003-big.md").write_text(
        "\n\n".join(f"# gap {i}\n\n" + "gap " * 200 for i in range(30)) + "\n")
    post_search(base, "gap", "a", cwd=str(repo))
    assert wait_until(lambda: "gap " in post_search(base, "gap", "b", cwd=str(repo)).split("docs：")[-1])
    out = post_search(base, "gap", "c", cwd=str(repo))
    assert len(out) <= DOC_LIMIT and out.split("docs：")[-1].count("\n- [") <= 3


def test_prompt_hook_sends_cwd(server):
    base, path = server
    r = subprocess.run([sys.executable, str(ROOT / "hooks" / "prompt.py"), path],
                       input=json.dumps({"prompt": "how was the stint gap handled last time", "session_id": "mine", "cwd": "/nowhere"}),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and r.stdout.startswith("memory：")


def test_docs_step_errors_do_not_lose_the_memory_block(docs_server, monkeypatch):
    base, tmp_path = docs_server
    repo = repo_with_docs(tmp_path)
    import docs as D
    monkeypatch.setattr(D, "repo_root", lambda cwd: (_ for _ in ()).throw(OSError("boom")))
    out = post_search(base, "stint gap", "err", cwd=str(repo))
    assert out.startswith("memory：和这句话相关的 1 条") and "docs：" not in out


def test_sections_are_marked_injected_only_after_the_reply_went_out(docs_server, monkeypatch):
    base, tmp_path = docs_server
    repo = repo_with_docs(tmp_path)
    post_search(base, "stint gap", "warm2", cwd=str(repo))
    assert wait_until(lambda: "docs：" in post_search(base, "stint gap", "probe2", cwd=str(repo)))
    real = serve.Handler.reply
    calls = []

    def flaky(self, code, body):
        calls.append(body)
        if len(calls) == 1:
            raise BrokenPipeError()  # 客户端等不及关了
        real(self, code, body)

    monkeypatch.setattr(serve.Handler, "reply", flaky)
    try:
        post_search(base, "stint gap", "late2", cwd=str(repo))
    except Exception:
        pass
    out = post_search(base, "stint gap", "late2", cwd=str(repo))
    assert "Gap rule" in out  # 第一次没送到，不算注过


def test_sweep_runs_only_when_the_config_turns_it_on():
    # 扫尾会把 Ollama 拉起来占 6.7 GB，不做写入半实验的时候不能有；live.toml 没有 [write] 就不扫
    assert serve.sweep_enabled({"db": "x"}) is False
    assert serve.sweep_enabled({"write": {}}) is False
    assert serve.sweep_enabled({"write": {"sweep": True}}) is True


def test_main_does_not_start_the_sweep_thread_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(embed, "Reranker", FakeReranker)
    started = []
    real_thread = threading.Thread

    class Recording(real_thread):
        def __init__(self, *a, **k):
            started.append(getattr(k.get("target"), "__name__", ""))
            super().__init__(*a, **k)

    monkeypatch.setattr(serve.threading, "Thread", Recording)
    monkeypatch.setattr(serve.ThreadingHTTPServer, "serve_forever", lambda self: None)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    serve.main(config(tmp_path, free_port()))
    assert "run" in started and "exit_when_idle" in started and "sweep_forever" not in started
