"""UserPromptSubmit hook：拿这句 prompt 当 query 问常驻服务，把结果打到 stdout（Claude Code 会当 context 注入）。
服务没起来就起一个，这一次不注入"""
import json
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_CHARS = 15
NOT_A_QUESTION = ("<task-notification", "<system-reminder")


def main(config_path: str) -> None:
    hook = json.load(sys.stdin)
    query = hook["prompt"].strip()
    if len(query) < MIN_CHARS or query.startswith(NOT_A_QUESTION):  # 后台任务的通知也走这个 hook，不是用户在问
        return
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    base = f"http://127.0.0.1:{cfg['serve']['port']}"
    try:
        alive = urllib.request.urlopen(f"{base}/health", timeout=1).read() == b"session-mem"
    except TimeoutError:  # 读超时：服务起来了，还在加载模型
        return
    except urllib.error.URLError as e:
        if not isinstance(e.reason, ConnectionRefusedError):  # 连接超时等等：同上
            return
        subprocess.Popen([sys.executable, str(ROOT / "serve.py"), config_path],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        return
    if not alive:
        return
    body = json.dumps({"query": query, "session_id": hook["session_id"], "cwd": hook.get("cwd", "")}).encode()
    req = urllib.request.Request(f"{base}/search", body, {"Content-Type": "application/json"})
    try:
        text = urllib.request.urlopen(req, timeout=12).read().decode()
    except (TimeoutError, urllib.error.URLError, ConnectionError):  # 服务忙：这一句不注入，别把 traceback 甩给用户
        return
    if text:
        print(text)


if __name__ == "__main__":
    main(sys.argv[1])
