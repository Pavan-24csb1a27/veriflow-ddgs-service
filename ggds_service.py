"""
ddgs_service.py
-----------------
A small, standalone FastAPI service whose only job is wrapping the Python
`ddgs` package and exposing it over HTTP, so the Node.js middleware
(which has no equivalent library) can call it for web search.

`ddgs` is actually a multi-engine meta-search library (DuckDuckGo, Bing,
Brave, Yahoo, Startpage, Mojeek, Yandex, Google, Wikipedia) — not just
DuckDuckGo — which is part of why it tends to be more resilient than a
single-engine scraper: if one backend blocks/rate-limits, ddgs can often
still return results from another.

Deploy this alongside (or as a second service within) your Render
project. It needs no GPU, no torch, no heavy dependencies — just FastAPI
and ddgs — so it's cheap and fast to run on a free/low tier.

Run locally:
    uvicorn ddgs_service:app --host 0.0.0.0 --port 8001

Environment variables:
    DDGS_SERVICE_SECRET  - shared secret the middleware must send, so this
                            endpoint isn't an open, anonymous search proxy
                            for anyone who finds the URL.
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header, status
from pydantic import BaseModel
from ddgs import DDGS

app = FastAPI(title="VeriFlow DDGS Search Service")

DDGS_SERVICE_SECRET = os.getenv("DDGS_SERVICE_SECRET")


class SearchRequest(BaseModel):
    query: str
    max_results: int = 3


class SearchResultItem(BaseModel):
    title: str
    url: str
    body: str


class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    note: str = ""


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest, x_service_secret: Optional[str] = Header(default=None)):
    if DDGS_SERVICE_SECRET and x_service_secret != DDGS_SERVICE_SECRET:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid service secret.")

    if not payload.query.strip():
        return SearchResponse(results=[], note="Empty query.")

    try:
        raw_results = DDGS().text(payload.query, max_results=payload.max_results)
    except Exception as e:
        # Don't raise a 500 for a search-engine hiccup — return an empty
        # result set with an explanation, same graceful-degrade pattern
        # used throughout the rest of the pipeline.
        return SearchResponse(results=[], note=f"Search failed: {e}")

    results = [
        SearchResultItem(
            title=r.get("title", ""),
            url=r.get("href", r.get("url", "")),
            body=r.get("body", ""),
        )
        for r in raw_results
    ]

    note = "" if results else "Query ran but returned no results."
    return SearchResponse(results=results, note=note)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ddgs_service:app", host="0.0.0.0", port=8001, reload=True)