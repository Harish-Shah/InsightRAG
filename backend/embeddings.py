"""Shared embedding model.

A single bge embedder is used by **both** ingestion and query so the document
and query vector spaces match - mixing models silently breaks retrieval.

We use LangChain's ``HuggingFaceBgeEmbeddings`` because it is bge-aware: it
auto-prepends the bge *query instruction*
("Represent this sentence for searching relevant passages:") to ``embed_query``
only, while documents are embedded as-is. The model is local/offline/free and
needs no API key, so this module is import-safe with no NVIDIA credentials.

The model (~130 MB ``BAAI/bge-small-en-v1.5``, 384-dim) is downloaded once on
first use, then cached on disk by sentence-transformers. We additionally cache a
process-wide singleton so the model is loaded into RAM only once.

Usage::

    from backend.embeddings import get_embeddings
    emb = get_embeddings()
    vec = emb.embed_query("total deposits")      # 384 floats, normalized
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from backend.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceBgeEmbeddings:
    """Return the process-wide singleton bge embedder (loads the model once).

    ``normalize_embeddings=True`` gives unit-length vectors so cosine similarity
    equals dot product - which is what ChromaDB's distance expects.
    """
    return HuggingFaceBgeEmbeddings(
        model_name=settings.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":  # Manual sanity check: python -m backend.embeddings
    emb = get_embeddings()
    q = emb.embed_query("What were total deposits?")
    d_related = emb.embed_documents(["Total deposits stood at 2,52,127 crore."])[0]
    d_unrelated = emb.embed_documents(["The cafeteria serves lunch at noon."])[0]

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))  # vectors are normalized

    print(f"embed_query dim:     {len(q)}")
    print(f"embed_documents dim: {len(d_related)}")
    print(f"cosine(query, related):   {cosine(q, d_related):.3f}")
    print(f"cosine(query, unrelated): {cosine(q, d_unrelated):.3f}")
    print(f"empty string dim:    {len(emb.embed_query(''))}")
