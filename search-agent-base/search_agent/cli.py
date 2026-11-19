import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from .agent import Agent
from .model import ChatModel, DemoModel
from .search import LocalSearch, WikipediaSearch, search_tool
from .tools import ToolRegistry


def main():
    parser = argparse.ArgumentParser(description="Minimal extensible search agent")
    parser.add_argument("question")
    parser.add_argument("--provider", choices=["local", "wikipedia"], default="local")
    parser.add_argument("--corpus", default="examples/corpus.jsonl")
    parser.add_argument("--demo", action="store_true", help="Offline scripted model, not real inference")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--model", help="Overrides LLM_MODEL")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output", help="Save answer, status and conversation as JSON")
    args = parser.parse_args()
    if args.demo and args.provider != "local":
        parser.error("--demo requires --provider local so it stays offline")
    if args.output and Path(args.output).exists():
        parser.error("Output already exists; choose a new --output path")
    try:
        provider = LocalSearch(args.corpus) if args.provider == "local" else WikipediaSearch()
        model = DemoModel() if args.demo else ChatModel(model=args.model, temperature=args.temperature)
        result = Agent(model, ToolRegistry([search_tool(provider)]), max_steps=args.max_steps).run(args.question)
        print(result.answer or f"No final answer (status={result.status}).")
        print(f"status={result.status}, model_calls={result.steps}", file=sys.stderr)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Refuse accidental overwrites of saved student experiments.
            with path.open("x", encoding="utf-8") as stream:
                json.dump(asdict(result), stream, ensure_ascii=False, indent=2)
        return 0 if result.status == "completed" else 2
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
