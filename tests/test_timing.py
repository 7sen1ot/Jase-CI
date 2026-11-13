"""时延统计与门禁回归, 不调用真实模型。"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "timing", Path(__file__).resolve().parents[1] / "scripts" / "timing_benchmark.py")
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class Response(io.BytesIO):
    def __init__(self, payload, content_type="application/json"):
        super().__init__(payload)
        self.headers = {"Content-Type": content_type}


class TimingTests(unittest.TestCase):
    def test_nonstreaming_response_has_no_ttft(self):
        response = Response(json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode())
        with patch.object(timing.urllib.request, "urlopen", return_value=response):
            ttft, total, _, text = timing.chat_request("http://local/v1", "", "test", "hi", True)
        self.assertIsNone(ttft)
        self.assertGreater(total, 0)
        self.assertEqual(text, "hello")

    def test_ttft_ignores_role_only_chunk(self):
        clock = [0]

        class Stream(Response):
            def __iter__(self):
                clock[0] = .1
                yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
                clock[0] = .4
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n'
                clock[0] = .5
                yield b'data: [DONE]\n'

        with patch.object(timing.urllib.request, "urlopen", return_value=Stream(b"", "text/event-stream")), \
                patch.object(timing.time, "perf_counter", side_effect=lambda: clock[0]):
            ttft, total, _, text = timing.chat_request("http://local/v1", "", "test", "hi", True)
        self.assertEqual((ttft, total, text), (400, 500, "hello"))

    def test_empty_response_is_not_a_latency_sample(self):
        for payload, content_type in [(b'{"choices":[]}', "application/json"),
                                      (b'data: [DONE]\n', "text/event-stream")]:
            with self.subTest(content_type=content_type), \
                    patch.object(timing.urllib.request, "urlopen", return_value=Response(payload, content_type)):
                with self.assertRaises(ValueError):
                    timing.chat_request("http://local/v1", "", "test", "hi", True)

    def test_truncated_response_is_not_a_latency_sample(self):
        payloads = [
            (b'{"choices":[{"message":{"content":"partial"},"finish_reason":"length"}]}',
             "application/json"),
            (b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
             b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n'
             b'data: [DONE]\n', "text/event-stream"),
        ]
        for payload, content_type in payloads:
            with self.subTest(content_type=content_type), \
                    patch.object(timing.urllib.request, "urlopen", return_value=Response(payload, content_type)):
                with self.assertRaisesRegex(ValueError, "截断"):
                    timing.chat_request("http://local/v1", "", "test", "hi", True)

    def test_max_tokens_reaches_request(self):
        response = Response(b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}')
        with patch.object(timing.urllib.request, "urlopen", return_value=response) as send:
            timing.chat_request("http://local/v1", "", "test", "hi", False, max_tokens=2048)
        self.assertEqual(json.loads(send.call_args.args[0].data)["max_tokens"], 2048)

    def run_benchmark(self, samples, budget=100, rounds=1, workloads=None):
        with tempfile.TemporaryDirectory() as outdir, \
                patch.dict(os.environ, {"ROUNDS": str(rounds), "COOLDOWN_MS": "0",
                                        "TIMING_BUDGET_MS": "1000", "GITHUB_STEP_SUMMARY": ""}), \
                patch.object(timing, "WORKLOADS", workloads if workloads is not None else [
                    {"name": "test", "prompt": "hi", "budget_ms": budget}]), \
                patch.object(timing, "chat_request", side_effect=samples), \
                patch.object(timing.time, "sleep"), \
                patch("sys.argv", ["timing", "--outdir", outdir]), \
                contextlib.redirect_stdout(io.StringIO()):
            code = timing.main()
            report = json.loads((Path(outdir) / "timing-report.json").read_text(encoding="utf-8"))
        return code, report

    def test_missing_round_fails_gate_despite_fast_success(self):
        code, report = self.run_benchmark(
            [(None, 10, 2, "ok"), OSError("down"), OSError("down")], rounds=2)
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["workloads"][0]["successful_rounds"], 1)

    def test_workload_budget_is_enforced(self):
        code, report = self.run_benchmark([(None, 150, 2, "ok")])
        self.assertEqual(code, 1)
        self.assertFalse(report["workloads"][0]["passed"])

    def test_successful_retry_passes_without_inventing_ttft(self):
        code, report = self.run_benchmark([OSError("transient"), (None, 10, 2, "ok")])
        self.assertEqual(code, 0)
        self.assertEqual(report["workloads"][0]["retries"], 1)
        self.assertEqual(report["workloads"][0]["ttft_ms"], {"avg": None, "p95": None})

    def test_failed_workload_cannot_hide_behind_successful_workload(self):
        code, _ = self.run_benchmark([(None, 10, 2, "ok"), OSError("down"), OSError("down")],
                                    workloads=[{"name": "a", "prompt": "hi"},
                                               {"name": "b", "prompt": "hi"}])
        self.assertEqual(code, 1)

    def test_no_workloads_fails_gate(self):
        code, _ = self.run_benchmark([], workloads=[])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
