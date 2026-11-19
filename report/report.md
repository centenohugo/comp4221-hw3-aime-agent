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

_TODO_

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
