"""Retrieval over the Chroma index (Task 3.2).

Two paths over the same store:
  * ``get_text_retriever()`` - the general top-k retriever used by the RAG chain.
  * ``retrieve_visuals(query)`` - a visual-targeted search filtered to
    table/image/figure chunks, returning (doc, score) so the chain's safety net
    can decide whether to attach a supporting visual.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from backend.config import settings
from backend.vectorstore import get_vectorstore

# Visual content types surfaced by retrieve_visuals (text chunks excluded).
VISUAL_TYPES = ["table", "image", "figure"]


def get_text_retriever() -> VectorStoreRetriever:
    """Top-k similarity retriever over all chunks (text + visuals)."""
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.TOP_K})


def retrieve_visuals(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Return visual chunks (table/image/figure) with relevance scores.

    Scores are 0-1 (higher = more relevant); the chain compares the top hit
    against ``settings.VISUAL_SIM_THRESHOLD`` for the safety-net attach.
    """
    k = k or settings.VISUAL_K
    return get_vectorstore().similarity_search_with_relevance_scores(
        query,
        k=k,
        filter={"content_type": {"$in": VISUAL_TYPES}},
    )


if __name__ == "__main__":  # Manual check: python -m backend.retriever
    q = "rural infrastructure development fund"
    print(f"text retriever (k={settings.TOP_K}):")
    for d in get_text_retriever().invoke(q)[:3]:
        m = d.metadata
        print(f"  {m['content_type']:6s} [{m['report_year']}, p.{m['page']}]")
    print(f"\nretrieve_visuals (k={settings.VISUAL_K}):")
    for d, s in retrieve_visuals(q):
        m = d.metadata
        print(f"  score={s:.3f} {m['content_type']:6s} [{m['report_year']}, p.{m['page']}] "
              f"asset={m.get('asset_path')}")
