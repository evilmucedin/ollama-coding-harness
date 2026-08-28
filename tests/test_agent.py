import unittest

from och.agent import Agent, extract_text_tool_calls
from och.client import ChatResponse
from och.config import Config
from och.tools import Tool


class FakeClient:
    """Scripted stand-in for OllamaClient."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, model, messages, tools=None, options=None, on_token=None):
        self.calls.append({"model": model, "messages": [dict(m) for m in messages]})
        resp = self.responses.pop(0)
        if on_token and resp.content:
            on_token(resp.content)
        return resp


def make_tools(log):
    def echo(text: str) -> str:
        log.append(text)
        return f"echo:{text}"

    def boom() -> str:
        raise RuntimeError("kaboom")

    return {
        "echo": Tool(
            name="echo",
            description="echo",
            parameters={"text": {"type": "string", "_required": True}},
            fn=echo,
        ),
        "danger": Tool(
            name="danger",
            description="mutating",
            parameters={},
            fn=lambda: "did it",
            mutating=True,
        ),
        "boom": Tool(name="boom", description="raises", parameters={}, fn=boom),
    }


def tool_call(name, args):
    return {"function": {"name": name, "arguments": args}}


class AgentTest(unittest.TestCase):
    def _agent(self, responses, log=None, approver=None, **cfg):
        config = Config(model="fake", **cfg)
        client = FakeClient(responses)
        agent = Agent(client, config, make_tools(log if log is not None else []), approver=approver)
        return agent, client

    def test_plain_answer(self):
        agent, client = self._agent([ChatResponse(content="hi there")])
        self.assertEqual(agent.run("hello"), "hi there")
        # system + user + assistant
        roles = [m["role"] for m in agent.messages]
        self.assertEqual(roles, ["system", "user", "assistant"])

    def test_tool_loop_executes_and_feeds_back(self):
        log = []
        agent, client = self._agent(
            [
                ChatResponse(tool_calls=[tool_call("echo", {"text": "ping"})]),
                ChatResponse(content="done"),
            ],
            log=log,
        )
        self.assertEqual(agent.run("go"), "done")
        self.assertEqual(log, ["ping"])
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(tool_msgs, [{"role": "tool", "tool_name": "echo", "content": "echo:ping"}])
        # second model call saw the tool result
        self.assertIn(
            "echo:ping",
            [m.get("content") for m in client.calls[1]["messages"]],
        )

    def test_string_encoded_arguments(self):
        log = []
        agent, _ = self._agent(
            [
                ChatResponse(tool_calls=[tool_call("echo", '{"text": "json-str"}')]),
                ChatResponse(content="ok"),
            ],
            log=log,
        )
        agent.run("go")
        self.assertEqual(log, ["json-str"])

    def test_unknown_tool_reports_error(self):
        agent, _ = self._agent(
            [ChatResponse(tool_calls=[tool_call("nope", {})]), ChatResponse(content="ok")]
        )
        agent.run("go")
        tool_msg = next(m for m in agent.messages if m["role"] == "tool")
        self.assertIn("unknown tool", tool_msg["content"])

    def test_tool_exception_is_captured(self):
        agent, _ = self._agent(
            [ChatResponse(tool_calls=[tool_call("boom", {})]), ChatResponse(content="ok")]
        )
        agent.run("go")
        tool_msg = next(m for m in agent.messages if m["role"] == "tool")
        self.assertIn("kaboom", tool_msg["content"])

    def test_bad_arguments_are_captured(self):
        agent, _ = self._agent(
            [ChatResponse(tool_calls=[tool_call("echo", {"bogus": 1})]), ChatResponse(content="ok")]
        )
        agent.run("go")
        tool_msg = next(m for m in agent.messages if m["role"] == "tool")
        self.assertIn("bad arguments", tool_msg["content"])

    def test_approver_denial_blocks_mutating_tool(self):
        denied = []
        agent, _ = self._agent(
            [ChatResponse(tool_calls=[tool_call("danger", {})]), ChatResponse(content="ok")],
            approver=lambda name, args: denied.append(name) and False,
        )
        agent.run("go")
        tool_msg = next(m for m in agent.messages if m["role"] == "tool")
        self.assertIn("declined", tool_msg["content"])
        self.assertEqual(denied, ["danger"])

    def test_auto_approve_skips_approver(self):
        agent, _ = self._agent(
            [ChatResponse(tool_calls=[tool_call("danger", {})]), ChatResponse(content="ok")],
            approver=lambda name, args: self.fail("approver should not be called"),
            auto_approve=True,
        )
        agent.run("go")
        tool_msg = next(m for m in agent.messages if m["role"] == "tool")
        self.assertEqual(tool_msg["content"], "did it")

    def test_max_iterations_bails_out(self):
        looping = [
            ChatResponse(tool_calls=[tool_call("echo", {"text": "again"})]) for _ in range(5)
        ]
        agent, _ = self._agent(looping, max_iterations=3)
        result = agent.run("go")
        self.assertIn("stopped after 3 iterations", result)


class ExtractTextToolCallsTest(unittest.TestCase):
    NAMES = {"read_file", "bash"}

    def test_bare_json(self):
        calls = extract_text_tool_calls(
            '{"name": "read_file", "arguments": {"path": "a.py"}}', self.NAMES
        )
        self.assertEqual(
            calls, [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]
        )

    def test_tool_call_tags(self):
        content = (
            'thinking...\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n'
            "</tool_call>"
        )
        calls = extract_text_tool_calls(content, self.NAMES)
        self.assertEqual(calls[0]["function"]["name"], "bash")

    def test_json_fence(self):
        content = 'Sure:\n```json\n{"name": "read_file", "arguments": {"path": "x"}}\n```'
        calls = extract_text_tool_calls(content, self.NAMES)
        self.assertEqual(calls[0]["function"]["arguments"], {"path": "x"})

    def test_multiple_calls(self):
        content = (
            '{"name": "read_file", "arguments": {"path": "a"}}\n'
            '{"name": "bash", "arguments": {"command": "ls"}}'
        )
        calls = extract_text_tool_calls(content, self.NAMES)
        self.assertEqual([c["function"]["name"] for c in calls], ["read_file", "bash"])

    def test_python_dict_literal_with_single_quotes(self):
        content = (
            "{\"name\": \"read_file\", \"arguments\": "
            "{\"path\": 'a.py', \"note\": 'return a + b\\n'}}"
        )
        calls = extract_text_tool_calls(content, self.NAMES)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["function"]["arguments"], {"path": "a.py", "note": "return a + b\n"}
        )

    def test_braces_inside_strings_do_not_confuse_scanner(self):
        content = '{"name": "bash", "arguments": {"command": "echo \'{ok}\'"}}'
        calls = extract_text_tool_calls(content, self.NAMES)
        self.assertEqual(calls[0]["function"]["arguments"], {"command": "echo '{ok}'"})

    def test_ignores_unknown_names_and_plain_json(self):
        self.assertEqual(
            extract_text_tool_calls('{"name": "nuke", "arguments": {}}', self.NAMES), []
        )
        self.assertEqual(
            extract_text_tool_calls('{"key": "value", "n": 1}', self.NAMES), []
        )
        self.assertEqual(extract_text_tool_calls("just prose, no json", self.NAMES), [])
        self.assertEqual(extract_text_tool_calls("", self.NAMES), [])

    def test_agent_uses_fallback_parser(self):
        log = []
        config = Config(model="fake")
        client = FakeClient(
            [
                ChatResponse(content='{"name": "echo", "arguments": {"text": "hi"}}'),
                ChatResponse(content="done"),
            ]
        )
        agent = Agent(client, config, make_tools(log))
        self.assertEqual(agent.run("go"), "done")
        self.assertEqual(log, ["hi"])


if __name__ == "__main__":
    unittest.main()
