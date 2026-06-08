"""Latency comparison: streaming (/api/chat/stream) vs. blocking (/api/chat).

Hits a running backend and reports, for one query:
  * STREAMING : time-to-first-token, streaming duration, total time, token count
  * BLOCKING  : total time (the user waits this long before seeing anything)
  * the perceived-latency improvement (blocking total vs. streaming TTFT).

Run from the project root with the backend up (uvicorn backend.main:app):
    python scripts/measure_streaming.py "What is the Rural Infrastructure Development Fund?"
    python scripts/measure_streaming.py            # uses a default query
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
DEFAULT_QUERY = "What is the Rural Infrastructure Development Fund?"


def _post(path: str, payload: dict):
    return urllib.request.urlopen(  # noqa: S310 - local, trusted URL
        urllib.request.Request(
            BASE + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        ),
        timeout=180,
    )


def measure_stream(query: str) -> dict:
    """Read the SSE stream, timing the first token and the final frame."""
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    citations = visuals = 0
    with _post("/api/chat/stream", {"message": query}) as resp:
        for raw in resp:                       # iterate the response line-by-line
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:].strip())
            if ev["type"] == "token":
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tokens += 1
            elif ev["type"] == "metadata":
                citations = len(ev.get("citations", []))
                visuals = len(ev.get("visuals", []))
            elif ev["type"] == "error":
                raise RuntimeError(ev.get("detail", "stream error"))
    total = time.perf_counter() - t0
    return {"ttft": ttft, "total": total, "tokens": tokens,
            "citations": citations, "visuals": visuals}


def measure_blocking(query: str) -> dict:
    t0 = time.perf_counter()
    with _post("/api/chat", {"message": query}) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    total = time.perf_counter() - t0
    return {"total": total, "citations": len(data.get("citations", [])),
            "visuals": len(data.get("visuals", []))}


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    print(f"Query: {query}\n")

    print("STREAMING  /api/chat/stream ...")
    s = measure_stream(query)
    print(f"  time-to-first-token : {s['ttft']:.2f}s")
    print(f"  streaming duration  : {s['total'] - (s['ttft'] or 0):.2f}s")
    print(f"  total time          : {s['total']:.2f}s")
    print(f"  tokens / citations / visuals : {s['tokens']} / {s['citations']} / {s['visuals']}\n")

    print("BLOCKING   /api/chat ...")
    b = measure_blocking(query)
    print(f"  total time (wait before any text) : {b['total']:.2f}s")
    print(f"  citations / visuals : {b['citations']} / {b['visuals']}\n")

    if s["ttft"]:
        delta = b["total"] - s["ttft"]
        pct = 100 * delta / b["total"] if b["total"] else 0
        print(f"Perceived-latency improvement: first text {delta:.2f}s sooner "
              f"({pct:.0f}% faster to first paint).")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
