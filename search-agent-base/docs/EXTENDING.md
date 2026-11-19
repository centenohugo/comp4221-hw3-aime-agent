# Student extension guide

The base is deliberately small. Start by running the offline demo and inspecting
the JSON trace, then make one change at a time and compare the results.

## Improve retrieval

- Replace token overlap with BM25 or embeddings using the same `search` method.
- Add query rewriting, query decomposition or several complementary searches.
- Add deduplication and reranking. Preserve source URLs when merging results.
- Add a provider for a real web search API; keep credentials in environment variables.
- Add a separate page-reading tool, with response-size and timeout limits, to
  move beyond snippets. Define which hosts/protocols it can access.

## Improve the agent

- Write a better evidence-grounded prompt and measure citation correctness.
- Detect repeated searches and stop unproductive loops.
- Add a context budget, caching, latency/token accounting or streaming output.
- Test empty results, ambiguous questions, tool errors and conflicting sources.

The current loop handles tool calls generically. A new tool should normally
require only a handler, a schema and registration. A new task may also need a
new system prompt, stopping rules and output format.

## Evaluate improvements

Create a small fixed question set with expected sources and answers. Compare
answer correctness, source support, tool calls and latency under the same model
and budget. Separate offline scripted tests from real model experiments.
The three example documents are a setup demonstration, not a benchmark.
