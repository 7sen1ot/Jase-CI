"""上游接入回归: 本地 HTTP 替身验证实际请求, 无外部 API/真实凭证。"""
import contextlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


class UpstreamTests(unittest.TestCase):
    def setUp(self):
        self.hits = []
        self.reply = lambda body: (200, {
            "content": [{"type": "text", "text": "查询完成: SO-1001 已发货"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.hits.append((self.path, self.headers, body))
                status, result = owner.reply(body)
                payload = json.dumps(result).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # 超时测试中客户端已主动断开。

            def log_message(self, *args):
                pass

        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.worker = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.worker.start()
        self.base = f"http://127.0.0.1:{self.upstream.server_port}"

    def tearDown(self):
        self.upstream.shutdown()
        self.upstream.server_close()
        self.worker.join()

    @contextlib.contextmanager
    def agent(self, **overrides):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("UPSTREAM_", "ANTHROPIC_", "DEEPSEEK_"))}
        env.update(PORT=str(port), MOCK_MODEL="0", UPSTREAM_GATEWAY="1",
                   ANTHROPIC_BASE_URL=self.base, ANTHROPIC_AUTH_TOKEN="test-token",
                   ANTHROPIC_DEFAULT_FABLE_MODEL="glm-test")
        env.update(overrides)
        proc = subprocess.Popen(["node", "agent/server.js"], cwd=ROOT, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            self.agent_base = f"http://127.0.0.1:{port}"
            for _ in range(100):
                if proc.poll() is not None:
                    self.fail(proc.stderr.read().decode())
                try:
                    with urllib.request.urlopen(self.agent_base + "/healthz", timeout=.2) as r:
                        self.assertEqual(json.load(r)["pid"], proc.pid)
                    break
                except OSError:
                    time.sleep(.05)
            else:
                self.fail("agent startup timeout")
            yield
        finally:
            proc.terminate()
            proc.wait(timeout=5)
            proc.stderr.close()

    def request(self, gateway=True):
        path = "/v1/upstream/chat/completions" if gateway else "/v1/chat/completions"
        req = urllib.request.Request(self.agent_base + path,
                                     data=json.dumps({"messages": [
                                         {"role": "user", "content": "查询订单 SO-1001"}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_anthropic_aliases_bearer_and_tool_roundtrip(self):
        original_reply = self.reply

        def tool_reply(body):
            blocks = [b for m in body["messages"] for b in m["content"]]
            if any(b["type"] == "tool_result" for b in blocks):
                return original_reply(body)
            return 200, {"content": [{"type": "tool_use", "id": "call-1",
                                      "name": "get_order_status", "input": {"order_id": "SO-1001"}}],
                         "stop_reason": "tool_use", "usage": {"input_tokens": 20, "output_tokens": 7}}

        self.reply = tool_reply
        with self.agent(ANTHROPIC_BASE_URL=self.base + "/v1/", UPSTREAM_MAX_TOKENS="1024"):
            status, data = self.request(gateway=False)
        self.assertEqual(status, 200)
        self.assertIn("已发货", data["choices"][0]["message"]["content"])
        self.assertEqual(data["model"], "glm-test")
        self.assertEqual(data["usage"], {"prompt_tokens": 30, "completion_tokens": 12})
        self.assertEqual(len(self.hits), 2)
        for path, headers, body in self.hits:
            self.assertEqual(path, "/v1/messages")
            self.assertEqual(headers["Authorization"], "Bearer test-token")
            self.assertIsNone(headers.get("x-api-key"))
            self.assertEqual(headers["anthropic-version"], "2023-06-01")
            self.assertEqual(body["model"], "glm-test")
            self.assertEqual(body["max_tokens"], 1024)
            self.assertIn("ShopEase", body["system"])
            self.assertEqual(len(body["tools"]), 4)

    def test_explicit_upstream_settings_override_aliases(self):
        with self.agent(UPSTREAM_PROTOCOL="anthropic", UPSTREAM_BASE=self.base + "/",
                        UPSTREAM_MODEL="explicit-model", UPSTREAM_API_KEY="explicit-key",
                        ANTHROPIC_BASE_URL="http://127.0.0.1:1"):
            status, data = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(data["model"], "explicit-model")
        path, headers, body = self.hits[0]
        self.assertEqual(path, "/v1/messages")
        self.assertEqual(headers["x-api-key"], "explicit-key")
        self.assertIsNone(headers.get("Authorization"))

    def test_anthropic_api_key_and_model(self):
        with self.agent(ANTHROPIC_AUTH_TOKEN="", ANTHROPIC_API_KEY="api-key",
                        ANTHROPIC_MODEL="preferred-model"):
            status, data = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(data["model"], "preferred-model")
        self.assertEqual(self.hits[0][1]["x-api-key"], "api-key")

    def test_explicit_openai_protocol_ignores_anthropic_aliases(self):
        self.reply = lambda body: (200, {"choices": [{"message": {
            "role": "assistant", "content": "hello"}, "finish_reason": "length"}]})
        with self.agent(UPSTREAM_PROTOCOL="openai", UPSTREAM_BASE=self.base + "/v1/",
                        UPSTREAM_MODEL="openai-test", UPSTREAM_API_KEY="openai-key"):
            status, data = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(data["choices"][0]["finish_reason"], "length")
        path, headers, body = self.hits[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer openai-key")
        self.assertEqual(body["model"], "openai-test")

    def test_gateway_preserves_anthropic_stop_reasons(self):
        for reason, content, expected in [
            ("max_tokens", [{"type": "text", "text": "partial"}], "length"),
            ("tool_use", [{"type": "tool_use", "id": "call-1", "name": "lookup", "input": {}}], "tool_calls"),
        ]:
            with self.subTest(reason=reason):
                self.reply = lambda body: (200, {"content": content, "stop_reason": reason})
                with self.agent():
                    status, data = self.request()
                self.assertEqual(status, 200)
                self.assertEqual(data["choices"][0]["finish_reason"], expected)

    def test_thinking_only_response_is_not_success(self):
        self.reply = lambda body: (200, {"content": [{"type": "thinking", "thinking": "..."}],
                                         "stop_reason": "max_tokens"})
        with self.agent():
            status, data = self.request()
        self.assertEqual(status, 502)
        self.assertIn("finish_reason=length", data["error"]["message"])

    def test_agent_rejects_truncated_text_and_tool_calls(self):
        for content in [
            [{"type": "text", "text": "未完成的回答"}],
            [{"type": "tool_use", "id": "call-1", "name": "get_order_status",
              "input": {"order_id": "SO-1001"}}],
        ]:
            with self.subTest(content=content):
                self.hits.clear()
                self.reply = lambda body: (200, {"content": content, "stop_reason": "max_tokens"})
                with self.agent():
                    status, data = self.request(gateway=False)
                self.assertEqual(status, 502)
                self.assertIn("截断", data["error"]["message"])
                self.assertEqual(len(self.hits), 1)

    def test_tool_loop_exhaustion_is_not_success(self):
        self.reply = lambda body: (200, {
            "content": [{"type": "tool_use", "id": "call-1", "name": "get_order_status",
                         "input": {"order_id": "SO-1001"}}], "stop_reason": "tool_use"})
        with self.agent():
            status, data = self.request(gateway=False)
        self.assertEqual(status, 502)
        self.assertIn("工具调用轮次过多", data["error"]["message"])
        self.assertEqual(len(self.hits), 4)

    def test_timeout_is_bounded(self):
        original_reply = self.reply

        def slow_reply(body):
            time.sleep(.4)
            return original_reply(body)

        self.reply = slow_reply
        with self.agent(UPSTREAM_TIMEOUT_MS="100"):
            status, data = self.request()
        self.assertEqual(status, 502)
        self.assertIn("超时 (100ms)", data["error"]["message"])

    def test_upstream_error_redacts_credential(self):
        self.reply = lambda body: (401, {"error": "invalid credential: test-token"})
        with self.agent():
            status, data = self.request()
        self.assertEqual(status, 502)
        self.assertNotIn("test-token", json.dumps(data))
        self.assertIn("401", data["error"]["message"])


if __name__ == "__main__":
    unittest.main()
