"""ChromaDB vector store helper.

A single, persistent, local Chroma collection (``annual_reports``) backed by the
shared bge embedder. ``get_vectorstore()`` is the one entry point reused by the
ingestion populate step and the retriever so both talk to
the same store with the same embedding function.

Chroma only accepts metadata values of type str/int/float/bool (no ``None``, no
lists/dicts), so ``sanitize_metadata`` coerces our unit metadata before writing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma

from backend.config import settings
from backend.embeddings import get_embeddings


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce a metadata dict to Chroma-safe scalar values.

    - ``None`` -> "" (Chroma rejects None)
    - list/tuple/dict -> comma-joined / stringified
    - everything else kept if str/int/float/bool, else ``str()``-ed
    """
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, bool):  # bool before int (bool is an int subclass)
            clean[key] = value
        elif isinstance(value, (str, int, float)):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = ", ".join(str(v) for v in value)
        else:
            clean[key] = str(value)
    return clean


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Return the persistent Chroma vector store (process-wide singleton)."""
    settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.CHROMA_DIR),
    )


def collection_count() -> int:
    """Number of stored vectors in the collection."""
    return get_vectorstore()._collection.count()


if __name__ == "__main__":  # Manual round-trip: python -m backend.vectorstore
    vs = get_vectorstore()
    print(f"collection '{settings.COLLECTION_NAME}' count: {collection_count()}")
