import copy
import io
import json
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from search_agent import Agent, Tool, ToolRegistry
from search_agent.model import ChatModel, DemoModel
from search_agent.search import LocalSearch, WikipediaSearch, search_tool

ROOT = Path(__file__).resolve().parents[1]


def call(name="echo", arguments='{"value": "hello"}', id="c1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(copy.deepcopy((messages, tools)))
        return next(self.replies)


class AgentTests(unittest.TestCase):
    def registry(self):
        return ToolRegistry([Tool("echo", "Echo value", {}, lambda value: value)])

    def test_tool_result_reaches_model_with_matching_id(self):
        model = ScriptedModel([{"tool_calls": [call()]}, {"content": "done"}])
        result = Agent(model, self.registry()).run("question")
        self.assertEqual((result.status, result.steps, result.answer), ("completed", 2, "done"))
        observation = model.requests[1][0][-1]
        self.assertEqual(observation["tool_call_id"], "c1")
        self.assertEqual(json.loads(observation["content"]), {"ok": True, "result": "hello"})

    def test_multiple_calls_and_error_recovery(self):
        model = ScriptedModel([{"tool_calls": [call(arguments="broken", id="bad"), call(id="good")]},
                               {"content": "recovered"}])
        result = Agent(model, self.registry()).run("question")
        observations = [m for m in result.messages if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in observations], ["bad", "good"])
        self.assertFalse(json.loads(observations[0]["content"])["ok"])
        self.assertTrue(json.loads(observations[1]["content"])["ok"])

    def test_budget_does_not_return_intermediate_answer(self):
        model = ScriptedModel([{"content": "I will search", "tool_calls": [call()]}])
        result = Agent(model, self.registry(), max_steps=1).run("question")
        self.assertEqual((result.status, result.answer), ("max_steps", ""))
        self.assertEqual(len(model.requests), 1)

    def test_fresh_history_per_run(self):
        model = ScriptedModel([{"content": "a"}, {"content": "b"}])
        agent = Agent(model, self.registry())
        agent.run("first")
        agent.run("second")
        self.assertEqual(len(model.requests[1][0]), 2)
        self.assertNotIn("first", json.dumps(model.requests[1][0]))

    def test_truncated_and_empty_responses(self):
        for reply in [{"content": "partial", "finish_reason": "length"}, {"content": None}]:
            self.assertEqual(Agent(ScriptedModel([reply]), self.registry()).run("q").status, "incomplete")

    def test_registry_errors(self):
        registry = self.registry()
        for name, args in [("unknown", "{}"), ("echo", "[]"), ("echo", "{}"), ("echo", '{"extra":1}')]:
            self.assertFalse(json.loads(registry.execute(name, args))["ok"])
        with self.assertRaises(ValueError):
            ToolRegistry([registry.tools["echo"], registry.tools["echo"]])

    def test_duplicate_call_ids_rejected(self):
        with self.assertRaises(ValueError):
            Agent(ScriptedModel([{"tool_calls": [call(), call()]}]), self.registry()).run("q")

    def test_offline_search_end_to_end(self):
        provider = LocalSearch(ROOT / "examples/corpus.jsonl")
        agent = Agent(DemoModel(), ToolRegistry([search_tool(provider)]))
        result = agent.run("rational arithmetic")
        self.assertEqual(result.status, "completed")
        self.assertIn("https://docs.python.org/3/library/fractions.html", result.answer)
        self.assertEqual(provider.search("zzzzzz")["results"], [])
        for top_k in [0, 11, True, "3"]:
            with self.assertRaises(ValueError):
                provider.search("test", top_k)

    @patch("urllib.request.urlopen")
    def test_wikipedia_parsing(self, urlopen):
        urlopen.return_value = io.BytesIO(json.dumps({"query": {"search": [
            {"title": "Example", "pageid": 123, "snippet": '<span class="searchmatch">A</span> &amp; B'}]}}).encode())
        result = WikipediaSearch().search("example")["results"][0]
        self.assertEqual(result["snippet"], "A & B")
        self.assertEqual(result["url"], "https://en.wikipedia.org/?curid=123")

    @patch("urllib.request.urlopen")
    def test_http_payload(self, urlopen):
        urlopen.return_value = io.BytesIO(b'{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}')
        model = ChatModel(api_key="test-key", base_url="https://example.test/v1")
        reply = model.complete([{"role": "user", "content": "question"}], [])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertNotIn("tools", json.loads(request.data))
        self.assertEqual(reply["content"], "ok")

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_retry_transient_http_failure(self, urlopen, sleep):
        urlopen.side_effect = [urllib.error.HTTPError("https://example.test", 429, "rate", {}, None),
            io.BytesIO(b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}')]
        model = ChatModel(api_key="test-key")
        self.assertEqual(model.complete([], self.registry().schemas())["content"], "ok")
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(json.loads(urlopen.call_args.args[0].data)["tools"][0]["function"]["name"], "echo")

    @patch("urllib.request.urlopen")
    def test_auth_error_is_not_retried_or_leaked(self, urlopen):
        urlopen.side_effect = urllib.error.HTTPError("https://example.test", 401, "secret", {}, None)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as error:
            ChatModel(api_key="test-key").complete([], [])
        self.assertNotIn("secret", str(error.exception))
        self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
