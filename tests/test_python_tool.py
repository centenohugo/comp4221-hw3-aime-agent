import json
import os
import unittest
from unittest.mock import patch

from src import ToolRegistry
from src.python_tool import execute_python_code, python_tool, truncate


class PythonToolTests(unittest.TestCase):
    def test_stdout_and_return_value(self):
        out = execute_python_code("print('hi')\n2 + 3")
        self.assertTrue(out["success"])
        self.assertEqual(out["stdout"].strip(), "hi")
        self.assertEqual(out["return_value"], "5")
        self.assertIsNone(out["error"])

    def test_required_modules_available(self):
        code = ("from fractions import Fraction\nimport itertools, math, numpy as np, sympy as sp\n"
                "x = sp.symbols('x')\n"
                "print(sp.solve(x**2 - 4, x), math.comb(5, 2), Fraction(1, 3) + Fraction(1, 6),\n"
                "      len(list(itertools.permutations(range(3)))), int(np.arange(5).sum()))")
        out = execute_python_code(code)
        self.assertTrue(out["success"], out)
        self.assertEqual(out["stdout"].strip(), "[-2, 2] 10 1/2 6 10")

    def test_preloaded_modules(self):
        self.assertEqual(execute_python_code("sp.factorint(360)")["return_value"], "{2: 3, 3: 2, 5: 1}")

    def test_runtime_error_is_reported_not_raised(self):
        out = execute_python_code("print('before')\n1 / 0")
        self.assertFalse(out["success"])
        self.assertEqual(out["stdout"].strip(), "before")
        self.assertEqual(out["error"], "ZeroDivisionError: division by zero")
        self.assertIn('File "<code>", line 2', out["stderr"])
        self.assertNotIn("RUNNER", out["stderr"])

    def test_syntax_error(self):
        out = execute_python_code("def f(:\n  pass")
        self.assertFalse(out["success"])
        self.assertTrue(out["error"].startswith("SyntaxError"))

    def test_timeout(self):
        out = execute_python_code("while True: pass", timeout=3)
        self.assertTrue(out["timed_out"])
        self.assertFalse(out["success"])

    def test_no_state_between_calls(self):
        execute_python_code("secret_value = 42")
        self.assertFalse(execute_python_code("secret_value")["success"])

    def test_api_key_not_visible(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "sk-test"}):
            out = execute_python_code("import os\nos.environ.get('LLM_API_KEY')")
        self.assertTrue(out["success"])
        self.assertIsNone(out["return_value"])

    def test_sys_exit_is_reported(self):
        self.assertTrue(execute_python_code("raise SystemExit(3)")["error"].startswith("SystemExit"))

    def test_output_truncated(self):
        out = execute_python_code("print('x' * 50000)")
        self.assertLess(len(out["stdout"]), 5000)
        self.assertIn("characters truncated", out["stdout"])
        self.assertEqual(truncate("abc", 10), "abc")

    def test_invalid_input_through_registry(self):
        registry = ToolRegistry([python_tool()])
        self.assertEqual(registry.schemas()[0]["function"]["name"], "execute_python_code")
        for args in ['{"code": ""}', '{"code": 5}', "{}", '{"code": "1", "extra": 1}']:
            self.assertFalse(json.loads(registry.execute("execute_python_code", args))["ok"])
        ok = json.loads(registry.execute("execute_python_code", '{"code": "6 * 7"}'))
        self.assertEqual(ok["result"]["return_value"], "42")


if __name__ == "__main__":
    unittest.main()
