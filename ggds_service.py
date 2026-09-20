"""
ddgs_service.py
-----------------
A small, standalone FastAPI service wrapping `ddgs` with rate-limit resilience,
exponential backoff, backend rotation, and request caching.

Run locally:
    uvicorn ddgs_service:app --host 0.0.0.0 --port 8001

Environment variables:
    DDGS_SERVICE_SECRET - Shared secret header validation.
    DDGS_PROXY          - (Optional) Proxy URL (e.g., http://user:pass@proxy:8080 or socks5://...).
"""

import os
import time
import asyncio
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Header, status
from pydantic import BaseModel
from ddgs import DDGS

app = FastAPI(title="VeriFlow DDGS Search Service")

DDGS_SERVICE_SECRET = os.getenv("DDGS_SERVICE_SECRET")
DDGS_PROXY = os.getenv("DDGS_PROXY")  # Optional proxy string

# Configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.5
CACHE_TTL_SECONDS = 300  # 5-minute cache for duplicate queries

# Simple in-memory response cache: { query_lower: (timestamp, results_list) }
_query_cache: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}


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


def clean_cache():
    """Remove expired items from in-memory cache."""
    now = time.time()
    expired_keys = [k for k, (ts, _) in _query_cache.items() if now - ts > CACHE_TTL_SECONDS]
    for k in expired_keys:
        del _query_cache[k]


def perform_ddgs_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Executes search with backend fallback rotation across 'auto', 'lite', and 'html'.
    """
    backends = ["auto", "lite", "html"]
    last_error = None

    for backend in backends:
        try:
            ddgs_kwargs = {}
            if DDGS_PROXY:
                ddgs_kwargs["proxy"] = DDGS_PROXY

            with DDGS(**ddgs_kwargs) as ddgs:
                results = list(
                    ddgs.text(
                        keywords=query,
                        backend=backend,
                        max_results=max_results
                    )
                )
                if results:
                    return results
        except Exception as e:
            last_error = e
            # Continue to next backend if available
            continue

    if last_error:
        raise last_error
    return []


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest, x_service_secret: Optional[str] = Header(default=None)):
    if DDGS_SERVICE_SECRET and x_service_secret != DDGS_SERVICE_SECRET:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid service secret.")

    clean_query = payload.query.strip()
    if not clean_query:
        return SearchResponse(results=[], note="Empty query.")

    cache_key = f"{clean_query.lower()}:{payload.max_results}"
    
    # 1. Check in-memory cache
    clean_cache()
    if cache_key in _query_cache:
        _, cached_results = _query_cache[cache_key]
        return SearchResponse(
            results=[SearchResultItem(**r) for r in cached_results],
            note="Cached result."
        )

    # 2. Retry loop with Exponential Backoff
    raw_results = []
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Offload synchronous/blocking DDGS execution to a threadpool
            raw_results = await asyncio.to_thread(perform_ddgs_search, clean_query, payload.max_results)
            break  # Success
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                # Calculate sleep duration: 1.5s, 3.0s, etc.
                sleep_time = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
                await asyncio.sleep(sleep_time)

    if last_exception and not raw_results:
        return SearchResponse(results=[], note=f"Search failed after {MAX_RETRIES} attempts: {last_exception}")

    # 3. Format output
    results = [
        SearchResultItem(
            title=r.get("title", ""),
            url=r.get("href", r.get("url", "")),
            body=r.get("body", ""),
        )
        for r in raw_results
    ]

    # Save to cache if results were returned
    if results:
        _query_cache[cache_key] = (time.time(), [r.model_dump() for r in results])

    note = "" if results else "Query ran but returned no results."
    return SearchResponse(results=results, note=note)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ddgs_service:app", host="0.0.0.0", port=8001, reload=True)