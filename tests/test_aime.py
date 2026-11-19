import json
import unittest

from src.aime import (SYSTEM_PROMPT_NO_TOOL, SYSTEM_PROMPT_TOOL, format_trace, make_agent,
                      summarize)


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def complete(self, messages, tools, tool_choice="auto"):
        self.calls.append((json.loads(json.dumps(messages)), tools, tool_choice))
        return next(self.replies)


def code_call(code, id):
    return {"id": id, "type": "function",
            "function": {"name": "execute_python_code", "arguments": json.dumps({"code": code})}}


class AimeAgentTests(unittest.TestCase):
    def test_no_tool_setting(self):
        model = ScriptedModel([{"content": "So the answer is \\boxed{204}."}])
        agent = make_agent(model, use_tool=False)
        self.assertEqual(agent.max_steps, 1)
        result = agent.run("problem")
        messages, tools, _ = model.calls[0]
        self.assertEqual(tools, [])  # no tool schemas sent
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT_NO_TOOL})
        self.assertEqual(summarize(result), {"status": "completed", "steps": 1, "tool_calls": 0})

    def test_tool_setting_runs_python_and_tracks_steps(self):
        model = ScriptedModel([
            {"content": "Let me compute.", "tool_calls": [code_call("print(6 * 7)", "c1")]},
            {"content": "The answer is \\boxed{42}."}])
        agent = make_agent(model, use_tool=True)
        self.assertEqual(agent.max_steps, 20)
        result = agent.run("problem")
        _, tools, _ = model.calls[0]
        self.assertEqual([t["function"]["name"] for t in tools], ["execute_python_code"])
        self.assertEqual(model.calls[0][0][0]["content"], SYSTEM_PROMPT_TOOL)
        observation = json.loads(result.messages[3]["content"])
        self.assertEqual(observation["result"]["stdout"].strip(), "42")
        self.assertEqual(summarize(result), {"status": "completed", "steps": 2, "tool_calls": 1})
        trace = format_trace(result.messages)
        self.assertIn("step 1", trace)
        self.assertIn("print(6 * 7)", trace)

    def test_prompt_requirements(self):
        for phrase in ["AIME", "execute_python_code", "\\boxed{}", "step by step",
                       "math, fractions, itertools, sympy, numpy"]:
            self.assertIn(phrase, SYSTEM_PROMPT_TOOL)


if __name__ == "__main__":
    unittest.main()
