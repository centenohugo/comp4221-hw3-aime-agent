# Search Agent Base

A minimal search agent with a tool-calling loop, local document search and optional Wikipedia search. Requires Python 3.10+; no third-party dependencies.

## Run

Run commands from this directory. Try the offline demo first (scripted responses, no API key needed):

```bash
python3 -m search_agent "Python rational arithmetic" --demo
```

To use a real model, set your API credentials and the model provided by your instructor:

```bash
export LLM_API_KEY="your-key"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"

python3 -m search_agent "Which Python module supports rational arithmetic?"
python3 -m search_agent "What is the Euclidean algorithm?" --provider wikipedia
```

The default search uses `examples/corpus.jsonl`. Wikipedia search requires internet access. Environment variables must be set in your shell; `.env` files are not loaded automatically.

## Options

- `--corpus path/to/documents.jsonl`: use a local corpus; each line needs `title`, `text` and `url` string fields.
- `--max-steps 8`: maximum number of model calls.
- `--output results/run.json`: save the answer and conversation; use a new filename for each run.
- `--model MODEL`: override `LLM_MODEL`.

## Extend

- `search_agent/agent.py`: agent loop and system prompt.
- `search_agent/search.py`: search providers.
- `search_agent/tools.py`: tool interface and registry.
- `search_agent/model.py`: model API client.
- `examples/custom_tool.py`: example of adding a tool.

## Test

```bash
python3 -m unittest discover -s tests -v
```
