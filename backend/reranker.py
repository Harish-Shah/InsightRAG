"""Reranking stage between Chroma retrieval and the LLM (retrieval-quality upgrade).

The bi-encoder embedding (``bge-small``) scores a query and a passage independently,
which is fast but coarse. A cross-encoder instead reads the ``(query, passage)`` pair
jointly, so it ranks the chunk that actually answers the question above merely
on-topic ones. This module retrieves a wide candidate set, rescores it, and keeps
only the best ``RERANK_TOP_N`` for the LLM - the upgrade IMPLEMENTATION_PLAN §12 calls
out ("``bge-small`` + no reranker set a retrieval-quality ceiling").

Two interchangeable backends, both exposing LangChain's ``BaseDocumentCompressor``
seam, so either drops into ``ContextualCompressionRetriever`` with no other changes:

  * ``nvidia`` (default) - hosted NVIDIA NeMo Retriever reranking NIM
    (:class:`NVIDIARerank`), reusing ``NVIDIA_API_KEY``; no local model/CPU cost.
  * ``local`` - a sentence-transformers cross-encoder (e.g. ``BAAI/bge-reranker-base``)
    wrapped by ``HuggingFaceCrossEncoder``; fully offline, needs no key.

Whichever backend is active is wrapped in :class:`_ObservableReranker`, which adds
observability (candidate/result counts, latency, top scores) and a hard guarantee:
on **any** error it returns the candidates untouched (dense-similarity order) so a
turn never breaks because of reranking. Reranking only *reorders/slices the original*
``Document`` objects, so every provenance field (citations, visual metadata) survives.

Swapping rerankers is config-only - change ``RERANK_BACKEND`` / ``RERANK_MODEL`` in
``.env``. To add a new engine, implement ``BaseDocumentCompressor`` and return it from
:func:`_build_inner`.

Usage::

    from backend.reranker import get_reranker, rerank
    top = rerank("total deposits and borrowings", candidate_docs)   # list[Document]
"""

from __future__ import annotations

import logging
import operator
import time
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Optional

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict

from backend.config import settings

log = logging.getLogger("rag.reranker")

# Metadata key holding each doc's reranker score. NVIDIARerank already writes this
# name; the local compressor mirrors it so logging + the eval script are uniform.
SCORE_KEY = "relevance_score"


class _LocalCrossEncoderCompressor(BaseDocumentCompressor):
    """A local sentence-transformers cross-encoder as a LangChain compressor.

    Scores every ``(query, page_content)`` pair, sorts descending, keeps the top
    ``top_n``, and records the score in ``metadata[SCORE_KEY]``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    encoder: Any = None          # langchain_community HuggingFaceCrossEncoder
    top_n: int = 5

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        scores = self.encoder.score([(query, doc.page_content) for doc in documents])
        ranked = sorted(
            zip(documents, scores), key=operator.itemgetter(1), reverse=True
        )
        top: list[Document] = []
        for doc, score in ranked[: self.top_n]:
            doc.metadata[SCORE_KEY] = float(score)
            top.append(doc)
        return top


class _ObservableReranker(BaseDocumentCompressor):
    """Wrap a backend compressor with logging, latency, and a safe fallback.

    The fallback is the whole point: if scoring raises (model error, NIM 429,
    timeout, ...), we return the candidates in their incoming (similarity) order -
    which equals the legacy top-k behaviour - rather than failing the turn.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: BaseDocumentCompressor
    top_n: int = 5

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        n_in = len(documents)
        start = time.perf_counter()
        try:
            out = list(self.inner.compress_documents(documents, query, callbacks))
        except Exception as exc:  # noqa: BLE001 - reranking must never break a turn
            dur_ms = (time.perf_counter() - start) * 1000
            log.warning(
                "rerank failed after %.0fms; falling back to similarity order: %s",
                dur_ms, exc,
            )
            return list(documents)[: self.top_n]
        dur_ms = (time.perf_counter() - start) * 1000
        top_scores = [
            round(float(d.metadata[SCORE_KEY]), 3)
            for d in out[:3] if SCORE_KEY in d.metadata
        ]
        log.info("rerank: %d -> %d in %.0fms; top=%s", n_in, len(out), dur_ms, top_scores)
        return out


def _build_inner() -> BaseDocumentCompressor:
    """Construct the configured backend compressor (imports are lazy/per-backend)."""
    backend = settings.RERANK_BACKEND.strip().lower()
    if backend == "nvidia":
        from langchain_nvidia_ai_endpoints import NVIDIARerank

        kwargs: dict[str, Any] = {
            "model": settings.RERANK_MODEL,
            "api_key": settings.require_nvidia_api_key(),
            "top_n": settings.RERANK_TOP_N,
            "truncate": "END",   # guard long unsplit table/figure chunks (~3.8k chars)
        }
        if settings.RERANK_BASE_URL:                 # only for a self-hosted NIM
            kwargs["base_url"] = settings.RERANK_BASE_URL
        return NVIDIARerank(**kwargs)

    if backend == "local":
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        encoder = HuggingFaceCrossEncoder(model_name=settings.RERANK_MODEL)
        return _LocalCrossEncoderCompressor(encoder=encoder, top_n=settings.RERANK_TOP_N)

    raise ValueError(
        f"Unknown RERANK_BACKEND {settings.RERANK_BACKEND!r} "
        "(expected 'nvidia' or 'local')"
    )


@lru_cache(maxsize=1)
def get_reranker() -> BaseDocumentCompressor:
    """Return the process-wide reranker singleton (observable wrapper + backend).

    The backend loads lazily on first call (model download / client setup happens
    here). Construction errors propagate to the caller, which is expected to fall
    back to plain retrieval - see ``backend.retriever.get_text_retriever``.
    ``lru_cache`` does not cache exceptions, so a transient failure is retried next
    call.
    """
    inner = _build_inner()
    log.info(
        "reranker ready: backend=%s model=%s candidates=%d top_n=%d",
        settings.RERANK_BACKEND, settings.RERANK_MODEL,
        settings.RERANK_CANDIDATES, settings.RERANK_TOP_N,
    )
    return _ObservableReranker(inner=inner, top_n=settings.RERANK_TOP_N)


def rerank(query: str, documents: Sequence[Document]) -> list[Document]:
    """Convenience for scripts/tests: rerank ``documents`` against ``query``."""
    return list(get_reranker().compress_documents(documents, query))


if __name__ == "__main__":  # Manual check: python -m backend.reranker
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from backend.vectorstore import get_vectorstore

    q = "What is the Rural Infrastructure Development Fund?"
    cands = get_vectorstore().similarity_search(q, k=settings.RERANK_CANDIDATES)
    print(f"backend={settings.RERANK_BACKEND} model={settings.RERANK_MODEL}")
    print(f"candidates={len(cands)} -> top_n={settings.RERANK_TOP_N}\n")
    for d in rerank(q, cands):
        m = d.metadata
        score = m.get(SCORE_KEY)
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "  -  "
        print(f"  score={score_s} {m.get('content_type',''):6s} "
              f"[{m.get('report_year','?')}, p.{m.get('page','?')}] {m.get('chunk_id','')}")
