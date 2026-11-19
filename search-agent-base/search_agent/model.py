"""A small HTTP client for a Chat Completions compatible endpoint."""
import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class ChatModel:
    def __init__(self, api_key=None, base_url=None, model=None,
                 temperature=0.6, timeout=90, max_tokens=4096, retries=2):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        if not self.api_key:
            raise ValueError("Set LLM_API_KEY, or use --demo for an offline demonstration")
        parsed = urlsplit(self.base_url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise ValueError("Use HTTPS, or HTTP on localhost for a local model")
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.retries = retries

    def complete(self, messages, tools):
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "stream": False}
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json"}, method="POST")
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
                choice = data["choices"][0]
                return {**choice["message"], "finish_reason": choice.get("finish_reason", "stop")}
            except urllib.error.HTTPError as exc:
                # Do not print response bodies or request headers (may contain secrets).
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise RuntimeError(f"Model API HTTP {exc.code}; check endpoint, model, credentials and quota") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries:
                    raise RuntimeError("Model API network error or timeout") from None
            time.sleep(min(2 ** attempt, 4))


class DemoModel:
    """Scripted fixture, NOT an LLM. Exercises the real tool dispatch offline."""
    def complete(self, messages, tools):
        if messages[-1]["role"] == "user":
            return {"content": None, "tool_calls": [{"id": "demo-search-1", "type": "function",
                    "function": {"name": "search", "arguments": json.dumps({
                        "query": messages[-1]["content"], "top_k": 3})}}]}
        observation = json.loads(messages[-1]["content"])
        results = observation.get("result", {}).get("results", [])
        lines = ["OFFLINE DEMO: scripted output, not an LLM answer."]
        for item in results:
            lines.append(f"- {item['title']}: {item['snippet']} ({item['url']})")
        if not results:
            lines.append("No matching local documents.")
        return {"content": "\n".join(lines)}
