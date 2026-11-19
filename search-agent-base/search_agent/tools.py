"""Tool interface and defensive dispatch, independent of any particular tool."""
import json
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., object]

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters,
        }}


class ToolRegistry:
    def __init__(self, tools=()):
        self.tools = {}
        for tool in tools:
            if tool.name in self.tools:
                raise ValueError(f"Duplicate tool: {tool.name}")
            self.tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self.tools.values()]

    def execute(self, name: str, arguments: str) -> str:
        """Always return a JSON observation so the model can recover from errors.

        Schemas describe arguments to the model. Each handler must validate values;
        this deliberately small registry is not a full JSON Schema validator.
        """
        try:
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")
            args = json.loads(arguments)
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be a JSON object")
            result = self.tools[name].handler(**args)
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"ok": False, "error": {
                "type": type(exc).__name__, "message": str(exc)[:1000],
            }}, ensure_ascii=False)
