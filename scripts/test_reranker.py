"""Reranker evaluation CLI (manual quality check).

For a query, shows how the reranker reorders Chroma's candidate set:
  * the ORIGINAL dense-similarity ranking (with similarity scores),
  * the RERANKED ranking (with cross-encoder scores) + latency, and
  * a movement summary (where each reranked chunk sat in the dense ranking),
plus each document's provenance metadata - proving it survives reranking.

Run from the project root (venv active):
    python scripts/test_reranker.py "What is the Rural Infrastructure Development Fund?"
    python scripts/test_reranker.py            # uses a default query
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `backend` importable when run as a loose script (not `python -m`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings              # noqa: E402
from backend.reranker import SCORE_KEY, rerank    # noqa: E402
from backend.vectorstore import get_vectorstore   # noqa: E402

DEFAULT_QUERY = "What is the Rural Infrastructure Development Fund?"
SNIPPET = 72


def _row(rank: int, score: str, doc) -> str:
    m = doc.metadata
    cite = f"[{m.get('report_year', '?')}, p.{m.get('page', '?')}]"
    section = (m.get("section") or "-")[:22]
    asset = m.get("asset_path") or "-"
    snippet = " ".join(doc.page_content.split())[:SNIPPET]
    return (f"  {rank:>2}. {score:>7}  {m.get('content_type', '?'):6s} {cite:26s} "
            f"sec={section:22s} asset={asset}\n"
            f"         {snippet}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY

    if not settings.RERANK_ENABLED:
        print("NOTE: RERANK_ENABLED=false - the chain uses plain top-k, but this script "
              "still exercises the reranker directly.\n")
    print(f"Query     : {query}")
    print(f"Backend   : {settings.RERANK_BACKEND}    Model: {settings.RERANK_MODEL}")
    print(f"Candidates: {settings.RERANK_CANDIDATES}  ->  Top-N: {settings.RERANK_TOP_N}\n")

    # ORIGINAL: dense-similarity candidates (with relevance scores) -------------
    scored = get_vectorstore().similarity_search_with_relevance_scores(
        query, k=settings.RERANK_CANDIDATES
    )
    candidates = [doc for doc, _ in scored]
    print(f"== ORIGINAL ranking (dense similarity, {len(candidates)} candidates) ==")
    for i, (doc, sim) in enumerate(scored, 1):
        print(_row(i, f"{sim:.3f}", doc))

    # RERANKED: cross-encoder over the SAME candidates -------------------------
    t0 = time.perf_counter()
    reranked = rerank(query, candidates)
    dur_ms = (time.perf_counter() - t0) * 1000
    print(f"\n== RERANKED ranking (top {len(reranked)}, {dur_ms:.0f}ms) ==")
    for i, doc in enumerate(reranked, 1):
        score = doc.metadata.get(SCORE_KEY)
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
        print(_row(i, score_s, doc))

    # Movement: where each reranked chunk sat in the dense ranking --------------
    orig_ids = [d.metadata.get("chunk_id") for d in candidates]
    print("\n== Movement (reranked position <- original dense position) ==")
    for i, doc in enumerate(reranked, 1):
        cid = doc.metadata.get("chunk_id")
        was = orig_ids.index(cid) + 1 if cid in orig_ids else "?"
        print(f"  rerank #{i}  <-  dense #{was}   {cid}")


if __name__ == "__main__":
    main()
