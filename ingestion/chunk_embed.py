"""Chunking + embedding into ChromaDB (Tasks 2.2 + 2.4).

Reads the Phase-1 manifest (``data/metadata.json``), splits text units to fit the
bge 512-token window while keeping each table/image/figure as a single unit, then
embeds and upserts everything into the persistent Chroma collection.

Deterministic chunk ids make the load idempotent: re-running upserts the same ids
instead of duplicating. Run::

    python -m ingestion.chunk_embed            # populate from existing manifest
    python -m ingestion.chunk_embed --reset    # clear the collection first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402
from backend.vectorstore import get_vectorstore, sanitize_metadata  # noqa: E402

# --- Chunking config (bge window is 512 tokens; ~1800 chars ~= 450 tokens) ---
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 320
MIN_TRAILING_CHARS = 200          # merge a tiny final chunk into the previous one
ADD_BATCH = 256                   # docs per Chroma add() call
_METADATA_KEYS = ("content_type", "source_pdf", "report_year", "page",
                  "section", "caption", "asset_path")

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _unit_metadata(unit: dict, unit_id: str, chunk_index: int) -> dict:
    """Build Chroma-safe metadata carrying provenance for citations/visuals."""
    meta = {k: unit.get(k) for k in _METADATA_KEYS}
    meta["unit_id"] = unit_id
    meta["chunk_index"] = chunk_index
    return sanitize_metadata(meta)


def _split_text(text: str) -> list[str]:
    """Split body text; merge a tiny trailing chunk into the previous one."""
    parts = [p for p in _splitter.split_text(text) if p.strip()]
    if len(parts) >= 2 and len(parts[-1]) < MIN_TRAILING_CHARS:
        parts[-2] = parts[-2] + "\n" + parts[-1]
        parts.pop()
    return parts or ([text] if text.strip() else [])


def chunk_units(units: list[dict]) -> list[Document]:
    """Manifest units -> LangChain Documents with deterministic ids in metadata.

    Text units split into multiple chunks (``{id}#{i}``); table/image/figure
    units pass through as exactly one chunk (keeping their own id).
    """
    docs: list[Document] = []
    for unit in units:
        unit_id = unit["id"]
        text = (unit.get("text") or "").strip()
        if unit["content_type"] == "text":
            for i, part in enumerate(_split_text(text)):
                meta = _unit_metadata(unit, unit_id, i)
                meta["chunk_id"] = f"{unit_id}#{i}"
                docs.append(Document(page_content=part, metadata=meta))
        else:
            if not text:
                # keep visual retrievable even if description is empty
                text = unit.get("caption") or f"{unit['content_type']} on page {unit.get('page')}"
            meta = _unit_metadata(unit, unit_id, 0)
            meta["chunk_id"] = unit_id
            docs.append(Document(page_content=text, metadata=meta))
    return docs


def load_units(manifest_path: Optional[Path] = None) -> list[dict]:
    path = manifest_path or settings.METADATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run `python -m ingestion.extract` first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["units"]


def populate(reset: bool = False, manifest_path: Optional[Path] = None) -> int:
    """Chunk the manifest and upsert all chunks into Chroma. Returns chunk count."""
    t0 = time.perf_counter()
    units = load_units(manifest_path)
    docs = chunk_units(units)
    ids = [d.metadata["chunk_id"] for d in docs]

    by_type = Counter(d.metadata["content_type"] for d in docs)
    lengths = [len(d.page_content) for d in docs]
    print(f"Chunked {len(units)} units -> {len(docs)} chunks {dict(by_type)}")
    if lengths:
        over = sum(1 for n in lengths if n > CHUNK_SIZE + 50)
        print(f"  chunk chars: min={min(lengths)} max={max(lengths)} "
              f"avg={sum(lengths)//len(lengths)} (>{CHUNK_SIZE}: {over})")

    vs = get_vectorstore()
    if reset:
        print("  reset: clearing existing collection ...")
        try:
            vs.delete_collection()
        except Exception as exc:  # noqa: BLE001
            print(f"  (delete_collection: {exc})")
        get_vectorstore.cache_clear()
        vs = get_vectorstore()

    total = len(docs)
    for start in range(0, total, ADD_BATCH):
        batch = docs[start:start + ADD_BATCH]
        batch_ids = ids[start:start + ADD_BATCH]
        vs.add_documents(documents=batch, ids=batch_ids)  # upsert by id
        print(f"  embedded {min(start + ADD_BATCH, total)}/{total}", flush=True)

    count = vs._collection.count()
    print(f"Done in {time.perf_counter() - t0:.1f}s. Collection count = {count}")
    return count


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Chunk metadata.json + embed into Chroma")
    ap.add_argument("--reset", action="store_true", help="clear the collection first")
    ap.add_argument("--manifest", help="manifest path (default: settings.METADATA_PATH)")
    args = ap.parse_args(argv)
    manifest = Path(args.manifest).resolve() if args.manifest else None
    populate(reset=args.reset, manifest_path=manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
