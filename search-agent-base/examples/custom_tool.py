"""A minimal tool extension. Run from the repository root; no model call needed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_agent import Tool, ToolRegistry


def word_count(text: str):
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return {"words": len(text.split())}


tool = Tool(
    name="word_count",
    description="Count whitespace-separated words in text.",
    parameters={"type": "object", "properties": {"text": {"type": "string"}},
                "required": ["text"], "additionalProperties": False},
    handler=word_count,
)
registry = ToolRegistry([tool])
print(registry.execute("word_count", '{"text": "A new tool"}'))
# To expose this tool to a model: Agent(model, registry, system_prompt="...").run(...)
