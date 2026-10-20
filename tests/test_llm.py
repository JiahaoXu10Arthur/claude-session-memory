import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import llm


def test_complete_posts_one_message_with_schema_and_no_thinking():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"choices": [{"message": {"content": '{"observations": []}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        m = llm.LLM({"base_url": f"http://127.0.0.1:{server.server_port}/v1", "model": "m"})
        assert m.complete("hi", {"type": "object"}) == '{"observations": []}'
    finally:
        server.shutdown()
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["model"] == "m"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert seen["body"]["reasoning_effort"] == "none"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["response_format"] == {"type": "json_schema", "json_schema": {"name": "observations", "schema": {"type": "object"}}}
