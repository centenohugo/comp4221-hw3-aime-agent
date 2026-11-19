"""AIME agent: system prompts, agent configuration and a single-problem CLI.

Reuses the baseline's Agent loop (src/agent.py) unchanged in structure; this module
only supplies the math system prompt, the tool set and the step budget for each
setting of the homework:

- no tool:   no tools, max_steps=1  (single-turn direct answer)
- with tool: execute_python_code, max_steps=20 (multi-step reasoning with tool calls)
"""
import argparse
import json
import sys
from pathlib import Path

from .agent import Agent, AgentResult
from .model import ChatModel
from .python_tool import python_tool
from .tools import ToolRegistry

NO_TOOL_MAX_STEPS = 1
TOOL_MAX_STEPS = 20

SYSTEM_PROMPT_NO_TOOL = """You are solving AIME (American Invitational Mathematics Examination) problems.

Reason step by step, check your work, and put your final answer in \\boxed{} format.
AIME answers are integers from 0 to 999."""

SYSTEM_PROMPT_TOOL = """You are solving AIME (American Invitational Mathematics Examination) problems.

Reason step by step. You have access to Python code execution via the execute_python_code tool. Use it to perform calculations, verify solutions, or explore patterns:
- Do exact arithmetic, algebra and counting in Python instead of by hand.
- Use brute force or enumeration to test hypotheses and check small cases.
- Verify your final answer with code before giving it.

Each tool call runs in a fresh Python process: no variables are kept between calls, so redefine what you need. Use print() to show results. If the code fails, read the error and fix it.

Available Python modules: math, fractions, itertools, sympy, numpy (already imported as math, fractions, Fraction, itertools, sympy/sp, numpy/np).

AIME answers are integers from 0 to 999. Put your final answer in \\boxed{} format."""


def make_agent(model, use_tool: bool, max_steps: int | None = None) -> Agent:
    """Build the agent for one of the two homework settings."""
    if use_tool:
        tools = ToolRegistry([python_tool()])
        prompt, default_steps = SYSTEM_PROMPT_TOOL, TOOL_MAX_STEPS
    else:
        tools = ToolRegistry([])  # no tools are sent to the model
        prompt, default_steps = SYSTEM_PROMPT_NO_TOOL, NO_TOOL_MAX_STEPS
    return Agent(model, tools, system_prompt=prompt, max_steps=max_steps or default_steps)


def count_tool_calls(messages: list[dict]) -> int:
    return sum(len(m.get("tool_calls") or []) for m in messages if m["role"] == "assistant")


def summarize(result: AgentResult) -> dict:
    """Reasoning-step statistics for one run (the full history is in result.messages)."""
    return {"status": result.status, "steps": result.steps,
            "tool_calls": count_tool_calls(result.messages)}


def format_trace(messages: list[dict], max_chars: int = 1500) -> str:
    """Human-readable trace of a conversation, for inspecting reasoning steps."""
    def clip(text):
        text = text or ""
        return text if len(text) <= max_chars else text[:max_chars] + f" ... [+{len(text) - max_chars} chars]"

    lines, step = [], 0
    for m in messages:
        if m["role"] == "assistant":
            step += 1
            lines.append(f"\n--- step {step}: assistant ---")
            if m.get("content"):
                lines.append(clip(m["content"]))
            for call in m.get("tool_calls") or []:
                try:
                    code = json.loads(call["function"]["arguments"]).get("code", "")
                except (ValueError, KeyError, AttributeError):
                    code = call.get("function", {}).get("arguments", "")
                lines.append(f"[tool call {call['function']['name']}]\n{clip(code)}")
        elif m["role"] == "tool":
            lines.append(f"[tool result]\n{clip(m['content'])}")
        elif m["role"] == "user" and step > 0:
            lines.append(f"\n--- user ---\n{clip(m['content'])}")
    return "\n".join(lines)


def load_problems(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Solve one AIME problem and print the trace")
    parser.add_argument("--id", type=int, required=True, help="Problem id in the dataset")
    parser.add_argument("--data", default="data/aime24.jsonl")
    parser.add_argument("--no-tool", action="store_true", help="Setting 1: no tool, 1 turn")
    parser.add_argument("--max-steps", type=int, help="Override the setting's step budget")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--model", help="Overrides LLM_MODEL")
    args = parser.parse_args()
    try:
        problems = {p["id"]: p for p in load_problems(args.data)}
        if args.id not in problems:
            parser.error(f"No problem with id {args.id} in {args.data}")
        problem = problems[args.id]
        model = ChatModel(model=args.model, temperature=args.temperature)
        agent = make_agent(model, use_tool=not args.no_tool, max_steps=args.max_steps)
        result = agent.run(problem["problem"])
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_trace(result.messages))
    print(f"\n=== gold answer: {problem['answer']} | {summarize(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
