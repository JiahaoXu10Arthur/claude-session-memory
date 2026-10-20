"""压缩用的模型：OpenAI 兼容的 HTTP 接口（Ollama 的 /v1）。只有这个文件知道请求长什么样"""
import json
import urllib.request

TIMEOUT = 900  # 40k 字符的窗口实测最长 234 秒


class LLM:
    def __init__(self, cfg: dict):
        self.base_url = cfg["base_url"]
        self.model = cfg["model"]

    def complete(self, prompt: str, schema: dict) -> str:
        """一条 user message，关 thinking，输出按 schema 约束成 JSON"""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "reasoning_effort": "none",
            "response_format": {"type": "json_schema", "json_schema": {"name": "observations", "schema": schema}},
        }).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]
