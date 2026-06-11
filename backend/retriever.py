"""Retrieval over the Chroma index.

Two paths over the same store:
  * ``get_text_retriever()`` - the general top-k retriever used by the RAG chain.
  * ``retrieve_visuals(query)`` - a visual-targeted search filtered to
    table/image/figure chunks, returning (doc, score) so the chain's safety net
    can decide whether to attach a supporting visual.
"""

from __future__ import annotations

import logging

from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from backend.config import settings
from backend.reranker import get_reranker
from backend.vectorstore import get_vectorstore

log = logging.getLogger("rag.retriever")

# Visual content types surfaced by retrieve_visuals (text chunks excluded).
VISUAL_TYPES = ["table", "image", "figure"]


def _plain_retriever(k: int) -> BaseRetriever:
    """Plain top-k similarity retriever over all chunks (text + visuals)."""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def get_text_retriever() -> BaseRetriever:
    """Retriever feeding the RAG chain.

    With reranking enabled (the default): pull ``RERANK_CANDIDATES`` from Chroma and
    wrap them in a cross-encoder reranker that returns the best ``RERANK_TOP_N``. The
    (condensed) query flows in via ``create_history_aware_retriever``, so the reranker
    scores against the rewritten question.

    With reranking disabled - or if the reranker cannot be built (no key / load error)
    - fall back to the plain top-``TOP_K`` similarity retriever, i.e. the legacy
    behaviour. Per-query scoring failures degrade inside the reranker itself.
    """
    if not settings.RERANK_ENABLED:
        return _plain_retriever(settings.TOP_K)

    if settings.RERANK_BACKEND.strip().lower() == "nvidia" and not settings.has_nvidia_key:
        log.warning("RERANK_BACKEND=nvidia but NVIDIA_API_KEY is not set; "
                    "reranking disabled, using plain top-%d.", settings.TOP_K)
        return _plain_retriever(settings.TOP_K)

    base = _plain_retriever(settings.RERANK_CANDIDATES)
    try:
        return ContextualCompressionRetriever(
            base_compressor=get_reranker(), base_retriever=base
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never block retrieval
        log.warning("reranker init failed (%s); using plain top-%d.",
                    exc, settings.TOP_K)
        return _plain_retriever(settings.TOP_K)


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
    mode = (f"rerank {settings.RERANK_CANDIDATES}->{settings.RERANK_TOP_N} "
            f"[{settings.RERANK_BACKEND}]") if settings.RERANK_ENABLED \
        else f"top-{settings.TOP_K}"
    print(f"text retriever ({mode}):")
    for d in get_text_retriever().invoke(q):
        m = d.metadata
        print(f"  {m['content_type']:6s} [{m['report_year']}, p.{m['page']}]")
    print(f"\nretrieve_visuals (k={settings.VISUAL_K}):")
    for d, s in retrieve_visuals(q):
        m = d.metadata
        print(f"  score={s:.3f} {m['content_type']:6s} [{m['report_year']}, p.{m['page']}] "
              f"asset={m.get('asset_path')}")
