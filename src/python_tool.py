"""Python code execution tool for the AIME agent.

Each call runs the model's code in a fresh, isolated Python subprocess:
- a separate process, so crashes, infinite loops or huge memory use cannot take
  down the agent (the process is killed after `timeout` seconds);
- `python -I` (isolated mode) and a minimal environment, so the code does not see
  the API key or other environment variables;
- a throwaway temporary working directory, deleted afterwards;
- stdout and stderr are captured and truncated so the observation stays small.

Like a notebook cell, the value of a final bare expression is returned as
`return_value`. Exceptions are reported as data (traceback in `stderr`, a one-line
summary in `error`), never raised, so the model can read the error and fix its code.

This is process isolation, not a security sandbox: the code runs with the user's
file and network permissions. That is acceptable for this homework, where the code
comes from the course's LLM solving math problems, but not for untrusted users.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .tools import Tool

MAX_CODE_CHARS = 20_000
MAX_OUTPUT_CHARS = 4_000
MAX_VALUE_CHARS = 2_000
MEMORY_LIMIT_BYTES = 2 * 1024 ** 3  # applied on POSIX only (Windows has no `resource`)

# Modules the README requires, pre-imported so a forgotten import is not an error.
PRELUDE = """\
import math, fractions, itertools, functools, collections
from fractions import Fraction
import numpy, numpy as np
import sympy, sympy as sp
"""

# Runs inside the child process. Reads the code from stdin, executes it and writes
# {"return_value", "error"} as JSON to the file given in argv[1].
RUNNER = r'''
import ast, json, linecache, sys, traceback
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
result_path = sys.argv[1]
source = sys.stdin.read()
# Register the source so tracebacks show the offending line of the model's code.
linecache.cache["<code>"] = (len(source), None, source.splitlines(True), "<code>")
namespace = {"__name__": "__main__"}
exec(sys.argv[2], namespace)
result = {"return_value": None, "error": None}
try:
    tree = ast.parse(source, "<code>", "exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<code>", "exec"), namespace)
    if last is not None:
        value = eval(compile(last, "<code>", "eval"), namespace)
        if value is not None:
            result["return_value"] = repr(value)
except BaseException as exc:
    # Hide the runner's own frames; show only frames from the model's code.
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename != "<code>":
        tb = tb.tb_next
    traceback.print_exception(type(exc), exc, tb)
    result["error"] = f"{type(exc).__name__}: {exc}"
sys.stdout.flush()
sys.stderr.flush()
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f)
'''


def truncate(text: str, limit: int) -> str:
    """Keep the head and tail of long text; the end usually holds the result or error."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters truncated] ...\n{text[-half:]}"


def _limit_memory():  # pragma: no cover - POSIX only
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


def _child_env() -> dict:
    # Only what Python needs to start; secrets such as LLM_API_KEY are not passed.
    keep = ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "TMPDIR", "LANG")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["OMP_NUM_THREADS"] = "1"
    return env


def execute_python_code(code: str, timeout: float = 30.0) -> dict:
    """Run `code` in a fresh subprocess and return its outputs as a dict."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a nonempty string")
    if len(code) > MAX_CODE_CHARS:
        raise ValueError(f"code must be at most {MAX_CODE_CHARS} characters")

    with tempfile.TemporaryDirectory(prefix="aime_py_", ignore_cleanup_errors=True) as workdir:
        result_path = Path(workdir) / "result.json"
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", RUNNER, str(result_path), PRELUDE],
                input=code, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout, cwd=workdir, env=_child_env(),
                preexec_fn=_limit_memory if os.name == "posix" else None)
        except subprocess.TimeoutExpired as exc:
            return {"success": False, "timed_out": True,
                    "stdout": truncate(_as_text(exc.stdout), MAX_OUTPUT_CHARS),
                    "stderr": truncate(_as_text(exc.stderr), MAX_OUTPUT_CHARS),
                    "return_value": None,
                    "error": f"TimeoutError: execution exceeded {timeout:g} seconds; "
                             "use a faster algorithm or a smaller search space"}

        try:
            child = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # The process died before reporting (e.g. killed for using too much memory).
            child = {"return_value": None,
                     "error": f"ProcessError: Python exited with code {proc.returncode}"}

    value = child.get("return_value")
    return {"success": child.get("error") is None, "timed_out": False,
            "stdout": truncate(proc.stdout, MAX_OUTPUT_CHARS),
            "stderr": truncate(proc.stderr, MAX_OUTPUT_CHARS),
            "return_value": truncate(value, MAX_VALUE_CHARS) if value is not None else None,
            "error": child.get("error")}


def _as_text(data) -> str:
    if data is None:
        return ""
    return data.decode("utf-8", "replace") if isinstance(data, bytes) else data


def python_tool(timeout: float = 30.0) -> Tool:
    """The `execute_python_code` tool, ready to register in a ToolRegistry."""
    def handler(code: str):
        return execute_python_code(code, timeout=timeout)

    return Tool(
        name="execute_python_code",
        description=(
            "Execute Python code to perform calculations, verify solutions and explore "
            "patterns. Each call runs in a fresh process (no state is kept between calls), "
            f"with a {timeout:g}-second time limit. Already imported: math, fractions "
            "(and Fraction), itertools, functools, collections, numpy (np), sympy (sp). "
            "Use print() to show results; the value of a final bare expression is also "
            "returned. Returns stdout, stderr, return_value and error."
        ),
        parameters={"type": "object", "properties": {
            "code": {"type": "string", "description": "Python code to execute"}},
            "required": ["code"], "additionalProperties": False},
        handler=handler,
    )
