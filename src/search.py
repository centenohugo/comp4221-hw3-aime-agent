"""Search providers: a small local baseline and optional live Wikipedia search."""
import html
import json
import math
import re
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from .tools import Tool


def validate(query, top_k):
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError("query must be a nonempty string of at most 2000 characters")
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer from 1 to 10")


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class LocalSearch:
    """English token-overlap baseline, normalized by document length.

    This is intentionally simple; students can replace it with BM25 or embeddings.
    """
    def __init__(self, corpus):
        self.documents = []
        for line in Path(corpus).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            if any(not isinstance(doc.get(k), str) for k in ("title", "text", "url")):
                raise ValueError("Each corpus row requires string title, text and url")
            self.documents.append(doc)
        self.counts = [Counter(tokens(d["title"] + " " + d["text"])) for d in self.documents]

    def search(self, query: str, top_k: int = 3):
        validate(query, top_k)
        terms = set(tokens(query))
        ranked = []
        for doc, counts in zip(self.documents, self.counts):
            score = sum(counts[t] for t in terms) / math.sqrt(max(1, sum(counts.values())))
            if score:
                ranked.append((score, doc))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return {"provider": "local", "results": [
            {"title": d["title"], "url": d["url"], "snippet": d["text"][:1500],
             "score": round(score, 4)} for score, d in ranked[:top_k]]}


class WikipediaSearch:
    """Search Wikipedia, not the entire web. No search API key required."""
    def search(self, query: str, top_k: int = 3):
        validate(query, top_k)
        params = urllib.parse.urlencode({"action": "query", "list": "search",
            "srsearch": query, "srlimit": top_k, "format": "json", "formatversion": 2})
        request = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + params,
            headers={"User-Agent": "COMP4221-SearchAgentBase/0.1 (educational search demo)"})
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.load(response)
        if "error" in data:
            raise RuntimeError("Wikipedia API error: " + str(data["error"].get("code", "unknown")))
        return {"provider": "wikipedia", "results": [
            {"title": row["title"], "url": "https://en.wikipedia.org/?curid=" + str(row["pageid"]),
             "snippet": html.unescape(re.sub(r"<[^>]+>", "", row.get("snippet", "")))[:1500]}
            for row in data["query"]["search"]]}


def search_tool(provider):
    return Tool("search", "Search for evidence. Returns titles, snippets and source URLs.",
        {"type": "object", "properties": {
            "query": {"type": "string", "description": "A focused search query"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3}},
         "required": ["query"], "additionalProperties": False}, provider.search)
