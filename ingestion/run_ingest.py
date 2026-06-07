"""Ingestion orchestrator (Task 2.5) - Milestone M1.

One command runs the whole offline pipeline end-to-end and supports clean
rebuilds and quick retrieval checks::

    python -m ingestion.run_ingest                     # extract (if needed) + embed
    python -m ingestion.run_ingest --reset             # clean rebuild from PDFs
    python -m ingestion.run_ingest --skip-extract      # reuse manifest, re-embed
    python -m ingestion.run_ingest --query "total deposits"   # retrieval check

Pipeline: PDFs -> extract (assets + metadata.json) -> chunk + embed -> Chroma.
Default reuses an existing ``metadata.json`` (extraction takes ~13 min); use
``--reset`` to rebuild everything or ``--skip-extract`` to force manifest reuse.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ₹ etc. on Windows
except Exception:
    pass

from backend.config import settings  # noqa: E402
from backend.vectorstore import get_vectorstore  # noqa: E402
from ingestion import extract as extract_mod  # noqa: E402
from ingestion import chunk_embed  # noqa: E402


def _clear_dir(path: Path, keep: set[str] = frozenset({".gitkeep"})) -> None:
    """Delete a directory's contents but keep placeholder files."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def run_query(query: str) -> int:
    vs = get_vectorstore()
    count = vs._collection.count()
    if count == 0:
        print("Collection is empty - run `python -m ingestion.run_ingest` first.")
        return 1
    print(f"Query: {query!r}  (collection size: {count})")
    print("-" * 70)
    results = vs.similarity_search_with_relevance_scores(query, k=settings.TOP_K)
    for doc, score in results:
        m = doc.metadata
        cite = f"[{m.get('report_year', '?')}, p.{m.get('page', '?')}]"
        asset = f" asset={m['asset_path']}" if m.get("asset_path") else ""
        preview = " ".join(doc.page_content.split())[:200]
        print(f"score={score:.3f}  {m.get('content_type'):6s} {cite}{asset}")
        print(f"    {preview}")
    return 0


def run_pipeline(reset: bool, skip_extract: bool) -> int:
    t0 = time.perf_counter()

    if reset:
        print("== RESET: clearing assets, manifest, and vector store ==")
        _clear_dir(settings.ASSETS_DIR)
        settings.METADATA_PATH.unlink(missing_ok=True)
        _clear_dir(settings.CHROMA_DIR)

    manifest_exists = settings.METADATA_PATH.exists()
    do_extract = not skip_extract and (reset or not manifest_exists)

    if do_extract:
        print("== EXTRACT: PDFs -> assets + metadata.json ==")
        rc = extract_mod.main([])
        if rc != 0:
            print("Extraction failed; aborting.")
            return rc
    else:
        reason = "--skip-extract" if skip_extract else "manifest already present"
        print(f"== EXTRACT: skipped ({reason}); reusing {settings.METADATA_PATH.name} ==")
        if not manifest_exists:
            print("No manifest to reuse; run without --skip-extract first.")
            return 1

    print("\n== EMBED: chunk + load into Chroma ==")
    count = chunk_embed.populate(reset=reset)

    print(f"\n== DONE in {time.perf_counter() - t0:.1f}s ==")
    print(f"Vector store: {settings.COLLECTION_NAME} @ {settings.CHROMA_DIR}")
    print(f"Chunks indexed: {count}")
    print('Try: python -m ingestion.run_ingest --query "total deposits"')
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the full ingestion pipeline (M1)")
    ap.add_argument("--reset", action="store_true",
                    help="clear assets + manifest + vector store, then rebuild")
    ap.add_argument("--skip-extract", action="store_true",
                    help="reuse existing metadata.json (do not re-run extraction)")
    ap.add_argument("--query", help="run a retrieval check against the vector store")
    args = ap.parse_args(argv)

    if args.query:
        return run_query(args.query)
    return run_pipeline(reset=args.reset, skip_extract=args.skip_extract)


if __name__ == "__main__":
    raise SystemExit(main())
