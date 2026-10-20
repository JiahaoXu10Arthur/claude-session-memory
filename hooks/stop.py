"""Stop hook：告诉常驻服务这个 session 的 transcript 有新内容了。服务没起就算了，下一句 prompt 的 hook 会起，扫尾接上"""
import json
import sys
import time
import tomllib
import urllib.request

SETTLE = 3  # 这一轮的 stop_hook_summary 行要等所有同步 Stop hook 跑完才写，等一下再通知服务


def main(config_path: str) -> None:
    hook = json.load(sys.stdin)
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    body = json.dumps({"transcript_path": hook["transcript_path"]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{cfg['serve']['port']}/write", body, {"Content-Type": "application/json"})
    time.sleep(SETTLE)
    try:
        urllib.request.urlopen(req, timeout=3)
    except OSError:  # 服务没起、超时、断开：都算了
        pass


if __name__ == "__main__":
    main(sys.argv[1])
