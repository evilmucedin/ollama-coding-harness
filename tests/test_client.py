import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from och.client import OllamaClient, OllamaError


class FakeOllamaHandler(BaseHTTPRequestHandler):
    """A tiny fake Ollama server for exercising the real HTTP path."""

    chat_chunks: list[dict] = []

    def log_message(self, *args):  # silence test output
        pass

    def do_GET(self):
        if self.path == "/api/tags":
            body = json.dumps(
                {"models": [{"name": "modelA:latest"}, {"name": "modelB:7b"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/api/chat":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for chunk in self.chat_chunks:
                self.wfile.write(json.dumps(chunk).encode() + b"\n")
        elif self.path == "/api/error":
            body = json.dumps({"error": "model not found"}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = OllamaClient(f"http://127.0.0.1:{cls.server.server_port}")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_list_models(self):
        self.assertEqual(self.client.list_models(), ["modelA:latest", "modelB:7b"])

    def test_chat_streams_content(self):
        FakeOllamaHandler.chat_chunks = [
            {"message": {"role": "assistant", "content": "Hel"}, "done": False},
            {"message": {"role": "assistant", "content": "lo"}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "eval_count": 5,
            },
        ]
        tokens = []
        resp = self.client.chat("m", [{"role": "user", "content": "hi"}], on_token=tokens.append)
        self.assertEqual(resp.content, "Hello")
        self.assertEqual(tokens, ["Hel", "lo"])
        self.assertEqual(resp.done_reason, "stop")
        self.assertEqual(resp.eval_count, 5)
        self.assertEqual(resp.message, {"role": "assistant", "content": "Hello"})

    def test_chat_collects_tool_calls(self):
        call = {"function": {"name": "bash", "arguments": {"command": "ls"}}}
        FakeOllamaHandler.chat_chunks = [
            {"message": {"role": "assistant", "content": "", "tool_calls": [call]}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ]
        resp = self.client.chat("m", [{"role": "user", "content": "list files"}])
        self.assertEqual(resp.tool_calls, [call])
        self.assertIn("tool_calls", resp.message)

    def test_chat_inline_error_chunk(self):
        FakeOllamaHandler.chat_chunks = [{"error": "boom"}]
        with self.assertRaisesRegex(OllamaError, "boom"):
            self.client.chat("m", [])

    def test_http_error_surfaces_detail(self):
        with self.assertRaisesRegex(OllamaError, "model not found"):
            with self.client._request("/api/error", {}):
                pass

    def test_unreachable_host(self):
        client = OllamaClient("http://127.0.0.1:1", timeout=2)
        with self.assertRaisesRegex(OllamaError, "Cannot reach Ollama"):
            client.list_models()

    def test_host_normalization(self):
        self.assertEqual(OllamaClient("localhost:11434").host, "http://localhost:11434")
        self.assertEqual(OllamaClient("http://x:1/").host, "http://x:1")


if __name__ == "__main__":
    unittest.main()
