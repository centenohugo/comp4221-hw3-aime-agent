# COMP 4221 Homework 3: AIME Math Problem Solver with a Python Tool

**Name:** Hugo Centeno Sanz
**Student ID:** 21431159

The agent is built on the provided search-agent baseline (`search-agent-base/`), copied into `src/` and modified there. The baseline's agent loop, model client and tool registry are reused unchanged unless stated in [Changes to the baseline](#changes-to-the-baseline).

---

## Step 1: Python Code Execution Tool

### Implementation

`src/python_tool.py` defines the tool `execute_python_code` with the baseline's `Tool` interface (the schema from the README plus a handler) and registers it in the baseline's `ToolRegistry`. It meets the four requirements as follows:

1. **Accept Python code as a string parameter.** The schema has one required string parameter, `code`. The handler rejects empty or overly long code.
2. **Execute the code in a safe environment.** Each call runs in a fresh Python subprocess with `python -I` (isolated mode), a 30-second timeout, a minimal environment that does not expose the API key, and a temporary working directory. A crash or infinite loop only affects the child process. `math`, `fractions`, `itertools`, `sympy` and `numpy` are pre-imported. This is process isolation, not a full security sandbox.
3. **Return the output (stdout, stderr and return value).** stdout and stderr are captured and truncated to 4000 characters. As in a notebook, the value of a final bare expression is returned as `return_value`.
4. **Handle errors gracefully.** Exceptions, syntax errors and timeouts are returned as data, never raised: the traceback, limited to the model's code, goes in `stderr` and a one-line summary in `error`. This lets the model read the error and fix its code on the next step.

### Core implementation

<!-- Screenshot: src/python_tool.py, RUNNER (line 43) and execute_python_code() (line 99) -->
_TODO: screenshot_

---

## Step 2: Agent Loop for AIME

### Implementation

The agent reuses the baseline's loop (`Agent.run` in `src/agent.py`). `src/aime.py` only configures it for AIME through `make_agent(model, use_tool)`:

1. **Initialize with a math system prompt.** Each run starts a fresh conversation with a system message and the problem as the user message. The prompt with the tool says the agent is solving AIME problems, asks it to reason step by step, encourages Python for exact calculations, enumeration and verification of the final answer, lists the available modules, notes that answers are integers from 0 to 999 and asks for the answer in `\boxed{}`.
2. **Loop for up to `max_steps`.** Each step is one model call: 20 with the tool and 1 without it. The loop ends when the model replies without tool calls, and that reply is the final answer. On the last step the model is asked to give its final answer and cannot call tools (see [Changes to the Baseline](#changes-to-the-baseline)).
3. **Support tool calling.** With the tool, the `ToolRegistry` contains `execute_python_code` and its schema is sent on every call. Each tool call the model makes is executed, and its JSON result is appended as a `tool` message with the matching `tool_call_id`, so the model sees the output on the next step. Without the tool, the registry is empty and no tools are sent.
4. **Track reasoning steps and conversation history.** `Agent.run` returns an `AgentResult` with the final answer, a status (`completed`, `incomplete` or `max_steps`), the number of steps and the full message history, including every tool call and tool result. `summarize()` reports the steps and the number of tool calls for each run, and `format_trace()` prints a readable trace (`python -m src.aime --id 60`).

---

## Step 3: Evaluation With and Without the Tool

_TODO_

| Setting | Tool | Max turns | Rollouts | Temperature | Accuracy |
|---|---|---|---|---|---|
| 1. Baseline | No | 1 | 4 | 0.6 | _TODO_ |
| 2. Python tool | Yes | 20 | 4 | 0.6 | _TODO_ |

---

## Changes to the Baseline

1. **Code location.** The baseline package `search_agent/` was copied into `src/` and modified there. Its tests were copied into `tests/`.
2. **`max_tokens` from 4096 to 8192** (`src/model.py`). 8192 is the maximum output length of `deepseek-chat`. AIME solutions, especially without the tool, can be long, and at 4096 tokens they risk being cut off before the final `\boxed{}` answer. The request timeout was raised from 90 s to 180 s to match.
3. **Forced final answer** (`src/agent.py`). In the baseline, if the agent used all of its `max_steps` on tool calls, it returned an empty answer, which is scored as wrong. Now, on the last allowed step of a multi-step run, the agent appends a short instruction to give the final answer in `\boxed{}` and calls the model with `tool_choice="none"`, so it must answer in text. No extra model call is made, so the run stays within `max_steps`. The single-turn setting without the tool is not affected.
4. **Environment.** The provided `pyproject.toml` was left unchanged. Instead of `uv sync`, a Python 3.12 environment was created with `uv venv` and `uv pip install sympy numpy math-verify`. Those are the only third-party packages needed, because the baseline uses only the standard library. The provided dependencies include packages for the search project (e.g. `pyserini`, which needs Java) that this homework does not use.
