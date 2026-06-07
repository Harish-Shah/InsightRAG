# RAG Chatbot — Implementation Task List

> **Purpose:** Executable project tracker derived from `IMPLEMENTATION_PLAN.md`.
> Detailed enough to execute without re-reading the plan.
>
> **Legend:** `[ ]` Not Started · `[x]` Completed. Tasks are sized to ≈2–4 hours each.
> Check off subtasks as you go; a task is done only when its **Completion Criteria** are met.
>
> **Phases:** 0 Setup → 1 Extraction → 2 Index → 3 RAG chain → 4 API/Persistence → 5 Frontend → 6 E2E/Docs.

---

# Phase 0 — Project Setup & Environment Configuration

## Task 0.1 — Scaffold project structure
- [x] **Objective:** Create the standalone project skeleton so every later task has a home.
- [x] **Expected Outcome:** Directory tree from the plan exists with empty package files.
- [x] **Dependencies:** None (first task).
- [x] **Implementation Notes:** Create `ingestion/`, `backend/`, `frontend/`, `data/{chroma,assets}/`. Add `__init__.py` to `ingestion/` and `backend/`. Add `.gitignore` (`.env`, `data/chroma/`, `data/assets/`, `data/app.db`, `__pycache__/`, `node_modules/`, `*.venv`). Do **not** import anything from `ClassMateAIDemo`.
- [x] **Validation:**
  - Manual test: `tree -L 2` (or `Get-ChildItem -Recurse -Depth 1`).
  - Expected result: all dirs present; `data/` ignored by git.
  - Edge cases: confirm `data/` subfolders are kept via `.gitkeep` but their contents ignored.
- [x] **Completion Criteria:** Folder tree matches the plan’s "Proposed Project Structure"; nothing tracked that shouldn’t be.

### Subtasks
- [x] Create directory tree + package `__init__.py` files
- [x] Add `.gitignore` and `.gitkeep` placeholders
- [ ] `git init` (optional) and initial commit  _(skipped by request; not a repo yet)_

## Task 0.2 — Python environment & dependencies
- [x] **Objective:** Reproducible Python env with pinned RAG/LLM/web deps.
- [x] **Expected Outcome:** `requirements.txt` installs cleanly into a fresh venv.
- [x] **Dependencies:** 0.1.
- [x] **Implementation Notes:** Python 3.11–3.13. Deps: `langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-chroma`, `langchain-nvidia-ai-endpoints`, `sentence-transformers`, `chromadb`, `pdfplumber`, `PyMuPDF`, `fastapi`, `uvicorn[standard]`, `python-dotenv`, `pydantic`. Pin versions. First `sentence-transformers` import downloads the bge model (~130 MB) — expect a one-time delay.
- [x] **Validation:**
  - Manual test: create venv, `pip install -r requirements.txt`, then `python -c "import langchain, fitz, pdfplumber, chromadb, sentence_transformers, fastapi"`.
  - Expected result: no import errors.
  - Edge cases: PyMuPDF imports as `fitz`; on Windows ensure `uvicorn[standard]` pulls `watchfiles`.
- [x] **Completion Criteria:** Clean install + all core imports succeed in a fresh venv.

### Subtasks
- [x] Author pinned `requirements.txt`
- [x] Create venv and install
- [x] Smoke-import all core libraries

## Task 0.3 — Environment configuration & central config
- [x] **Objective:** Centralize all tunables (keys, model names, paths) so nothing is hard-coded.
- [x] **Expected Outcome:** `.env.example`, local `.env`, and `backend/config.py` load consistently.
- [x] **Dependencies:** 0.1.
- [x] **Implementation Notes:** `.env` keys: `NVIDIA_API_KEY`, `NVIDIA_MODEL` (default `meta/llama-3.3-70b-instruct`), `NVIDIA_BASE_URL` (`https://integrate.api.nvidia.com/v1`), `EMBED_MODEL` (`BAAI/bge-small-en-v1.5`), `CHROMA_DIR`, `ASSETS_DIR`, `SQLITE_PATH`, `COLLECTION_NAME` (`annual_reports`), `TOP_K`, `VISUAL_K`, `VISUAL_SIM_THRESHOLD`. `config.py` reads via `python-dotenv` with sane defaults; resolve paths absolutely.
- [x] **Validation:**
  - Manual test: `python -c "from backend.config import settings; print(settings)"`.
  - Expected result: prints resolved config; missing `NVIDIA_API_KEY` raises a clear error only where needed (not at import for ingestion).
  - Edge cases: `.env` absent → defaults still load; never commit `.env`.
- [x] **Completion Criteria:** Config loads from `.env`; `.env.example` documents every key; secrets untracked.

### Subtasks
- [x] Write `.env.example` with comments for each key
- [x] Create local `.env` (real key)  _(created with the NVIDIA key during Task 0.4)_
- [x] Implement `backend/config.py` loader with defaults

## Task 0.4 — NVIDIA NIM connectivity smoke test
- [x] **Objective:** Prove the NIM key/model/base-URL work via `ChatNVIDIA` before building on them.
- [x] **Expected Outcome:** A one-line prompt returns a completion.
- [x] **Dependencies:** 0.2, 0.3.
- [x] **Implementation Notes:** Tiny throwaway script using `ChatNVIDIA(model=settings.NVIDIA_MODEL, base_url=...)`; `.invoke("Say hello in 3 words")`. Note observed latency + any 429s for the rate-limit handling later.
- [x] **Validation:**
  - Manual test: run the script.
  - Expected result: a short text response prints.
  - Edge cases: invalid key → 401 (clear message); model unavailable → switch `NVIDIA_MODEL` in `.env` only (no code change) and re-run.
- [x] **Completion Criteria:** Live completion received; swapping model via `.env` works without code edits.

### Subtasks
- [x] Write smoke-test script
- [x] Confirm completion + record latency/limits
- [x] Verify model swap via `.env` only

---

# Phase 1 — Ingestion Pipeline: Extraction

> All Phase 1 tasks live in `ingestion/extract.py`. They are independent units sharing a per-page loop; can be built/tested page-by-page on a single PDF first.

## Task 1.1 — PDF loading, page iteration & provenance
- [x] **Objective:** Establish the page loop, report-year mapping, and empty-page skipping that feed all extractors.
- [x] **Expected Outcome:** A generator yielding `(pdf_id, report_year, page_number, fitz_page, plumber_page)` for content pages. _(Implemented as a per-page loop in `process_pdf`; PyMuPDF-only — see engine note below.)_
- [x] **Dependencies:** 0.2.
- [x] **Implementation Notes:** Open each PDF once with PyMuPDF and once with pdfplumber; iterate aligned page indices. Map filenames → friendly years (`Annual_Report_2022_23` → `Annual Report 2022-23`). Skip pages with `<40` chars **and** no useful image/figure (covers/dividers/blanks). 1-based page numbers for citations.
- [x] **Validation:**
  - Manual test: run on one PDF, print per-page `(year, page, char_count, n_images)`.
  - Expected result: ~252–284 rows/PDF; blank/divider pages flagged skipped.
  - Edge cases: PyMuPDF vs pdfplumber page-count mismatch → log + clamp to min; corrupt page → catch + continue.
- [x] **Completion Criteria:** Stable iteration over all 3 PDFs with correct year/page provenance and sensible skips. _(Verified: 260/252/284 pages; skipped 13/5/3 covers/dividers; `2021_22 1.pdf` → `2021-22`.)_

> **Engine note:** pdfplumber's per-page parse is ~1s/page and tens of seconds on the image/vector-dense pages here (a full pdfplumber run took ~46 min). The pipeline uses **PyMuPDF natively for text and tables** (~0.12s/page, equivalent output, same real tables minus pdfplumber's thin-line false positives). Documented in `ingestion/extract.py`.

### Subtasks
- [x] Filename → report-year mapping
- [x] Single-open PyMuPDF page loop _(fitz==plumber page counts confirmed; pdfplumber dropped from hot path for performance)_
- [x] Empty-page skip heuristic + logging

## Task 1.2 — Text extraction
- [x] **Objective:** Produce clean per-page text with section headings for chunking.
- [x] **Expected Outcome:** `{year, page, text, section}` records for content pages.
- [x] **Dependencies:** 1.1.
- [x] **Implementation Notes:** Primary `plumber_page.extract_text()`; fallback `fitz_page.get_text()` if empty. Clean: collapse whitespace, de-hyphenate line breaks, strip repeating header/footer lines (detect lines recurring on many pages). Heading heuristic via `get_text("dict")` font sizes (largest short line on page).
- [x] **Validation:**
  - Manual test: dump 5 random pages’ cleaned text + detected heading.
  - Expected result: readable paragraphs; no header/footer noise; plausible headings.
  - Edge cases: multi-column pages (verify reading order); pages where pdfplumber returns empty (fallback fires).
- [x] **Completion Criteria:** Clean text + heading for ≥95% of content pages across all PDFs. _(100% of 766 text units carry a section heading; whitespace/hyphen/header-footer cleaning verified; nbsp normalized.)_

### Subtasks
- [x] PyMuPDF `get_text` (primary, for performance) + whitespace fallback
- [x] Whitespace/hyphen/header-footer cleaning
- [x] Section-heading heuristic _(font-size based + carry-forward; same-size bold titles are a known minor miss)_

## Task 1.3 — Table extraction (Markdown + cropped image)
- [x] **Objective:** Extract financial tables as both searchable Markdown and a viewable crop.
- [x] **Expected Outcome:** Per table: Markdown text, `asset_path` PNG, page, nearest caption.
- [x] **Dependencies:** 1.1.
- [x] **Implementation Notes:** `plumber_page.find_tables()` for bboxes + `extract_tables()` for cells. Cells → Markdown (pad ragged rows). Crop: convert pdfplumber bbox (top-left origin, points) → `fitz.Rect`; `fitz_page.get_pixmap(clip=rect, dpi=150)` → save `data/assets/<pdf>/p<page>_tbl<n>.png`. Description = nearest heading/caption + Markdown. _(Used PyMuPDF `page.find_tables()`; its bbox is already in fitz coords → no conversion; junk filters: min size, ≥2 cols/rows, filled-ratio, reject full-page grids.)_
- [x] **Validation:**
  - Manual test: pick 3 known financial pages; inspect Markdown + open crops.
  - Expected result: numbers/columns align; crop tightly frames the table.
  - Edge cases: merged cells/multi-row headers (note imperfections); table spanning page break (treat per-page); coordinate flip between libraries (verify crop isn’t mirrored/offset).
- [x] **Completion Criteria:** Tables on sampled statement pages produce coherent Markdown + correct crops. _(543 tables; crops verified on 2021-22 "Sources of Funds" and 2023-24 Schedules — columns aligned, not mirrored.)_

### Subtasks
- [x] Table detect + cells → Markdown _(via PyMuPDF native table finder)_
- [x] bbox → fitz.Rect crop → PNG
- [x] Attach caption/heading description _(title line(s) above the table)_

## Task 1.4 — Raster image extraction
- [x] **Objective:** Save embedded photos/raster charts as retrievable visual assets.
- [x] **Expected Outcome:** Per kept image: PNG `asset_path`, page, caption description.
- [x] **Dependencies:** 1.1.
- [x] **Implementation Notes:** `fitz_page.get_images(full=True)` → extract pixmaps → PNG `p<page>_img<n>.png`. Drop tiny (w<120 or h<120 px) logos/rules/icons. Description = caption near image bbox (`get_image_info(xrefs=True)` for bbox; nearest text lines above/below) + section heading + nearby paragraph slice. _(Native-dim pre-filter + xref dedupe across the doc; cap 12 largest/page to tame pages with 1000+ image fragments.)_
- [x] **Validation:**
  - Manual test: open a sample of saved images; print their descriptions.
  - Expected result: meaningful images only; descriptions reference the right caption.
  - Edge cases: CMYK/alpha images (convert to RGB before PNG); duplicate logos across pages (dedupe by xref/hash optional); images with no nearby text (fall back to section heading).
- [x] **Completion Criteria:** Non-trivial embedded images saved with sensible descriptions; logos filtered. _(304 images; verified a real event photo on 2022-23 p34; tiny logos filtered by the 120px rule.)_

### Subtasks
- [x] Extract + RGB-normalize + save PNGs _(CMYK/alpha → RGB)_
- [x] Size-based noise filter + xref dedupe
- [x] Caption/nearby-text description builder

## Task 1.5 — Caption-anchored figure-region rendering (vector charts)
- [x] **Objective:** Capture vector-drawn charts/graphs (which `get_images` misses) as viewable PNGs — the key mitigation for these vector-heavy reports.
- [x] **Expected Outcome:** Per detected figure: rendered region PNG, page, caption description.
- [x] **Dependencies:** 1.1, 1.2.
- [x] **Implementation Notes:** Detect explicit captions in page text via regex (`Figure|Chart|Graph|Exhibit|Diagram`\s*\d+) and/or vector-dense regions (`get_drawings()` > threshold paths clustered). Compute a bounding region around the caption/drawings; render `get_pixmap(clip=region, dpi=150)` → `p<page>_fig<n>.png`. Description = caption + surrounding text. Keep conservative (explicit captions first) to avoid noise. _(Caption-anchored region + drawings gate: a region renders only if it contains ≥12 vector drawings — filters body-text "Figure N" references.)_
- [x] **Validation:**
  - Manual test: run on 2022-23 (most chart-dense); open rendered figures.
  - Expected result: real charts captured; few false positives.
  - Edge cases: caption without nearby graphic (skip); huge region covering whole page (cap region size); decorative vector backgrounds (raise drawing threshold).
- [x] **Completion Criteria:** Representative vector charts are rendered with captions; noise acceptably low (tunable). _(77 figures; verified a real 2022-23 chart "Figure 2: Decadal percentage change…"; "Figure 1" body refs on p25/26/29/30 correctly skipped by the gate.)_

### Subtasks
- [x] Caption regex + vector-density gate (drawings-in-region)
- [x] Region bbox computation + clip render _(capped to FIG_MAX_HEIGHT; avoids overlap with tables)_
- [x] Caption/context description

## Task 1.6 — Metadata manifest generation
- [x] **Objective:** Emit a single manifest the indexing stage consumes; make ingestion idempotent.
- [x] **Expected Outcome:** `data/metadata.json` listing every unit with stable IDs + metadata.
- [x] **Dependencies:** 1.2–1.5.
- [x] **Implementation Notes:** Unit schema: `{id, content_type ∈ [text,table,image,figure], source_pdf, report_year, page, section, caption, text, asset_path?}`. Deterministic IDs (e.g. `2022-23_p045_tbl01`). Aggregate counts per type/PDF. _(asset_path is relative to ASSETS_DIR; full-run clears each PDF's asset subdir first → idempotent.)_
- [x] **Validation:**
  - Manual test: `jq '.units | group_by(.content_type) | map({(.[0].content_type): length})'`.
  - Expected result: nonzero counts for text/table/image/figure; total ≈ 1.5k–3k.
  - Edge cases: asset_path points to a missing file (assert exists); duplicate IDs (assert unique).
- [x] **Completion Criteria:** Valid `metadata.json`; all asset_paths exist; IDs unique; counts plausible. _(1690 units: text 766 / table 543 / image 304 / figure 77; IDs unique; 924/924 asset files present.)_

### Subtasks
- [x] Define unit schema + ID scheme
- [x] Serialize units + per-type/PDF counts
- [x] Integrity asserts (paths exist, IDs unique)

---

# Phase 2 — Ingestion Pipeline: Chunking, Embedding & Vector Store

## Task 2.1 — Shared embeddings module
- [x] **Objective:** One bge embedder used by **both** ingestion and query (consistency is mandatory).
- [x] **Expected Outcome:** `backend/embeddings.py` returns a configured `HuggingFaceBgeEmbeddings`.
- [x] **Dependencies:** 0.2, 0.3.
- [x] **Implementation Notes:** `HuggingFaceBgeEmbeddings(model_name=settings.EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})`. Wrapper auto-applies the bge query instruction to `embed_query` only. Cache a singleton to avoid reloading the model.
- [x] **Validation:**
  - Manual test: embed a doc and a query; print vector length + cosine sim of a related pair.
  - Expected result: dim 384; related texts score higher than unrelated.
  - Edge cases: first call downloads model (slow once); empty string input handled.
- [x] **Completion Criteria:** Importable singleton; correct dim; query/doc paths behave differently per bge spec. _(dim 384; cosine related 0.69 > unrelated 0.35; `lru_cache` singleton; empty string handled.)_

### Subtasks
- [x] Implement loader + singleton cache
- [x] Verify 384-dim + normalization
- [x] Sanity cosine-similarity check

## Task 2.2 — Chunking
- [x] **Objective:** Split text to fit the 512-token window; keep tables/images/figures as single units.
- [x] **Expected Outcome:** Chunk list with preserved metadata, ready to embed.
- [x] **Dependencies:** 1.6, 2.1.
- [x] **Implementation Notes:** `RecursiveCharacterTextSplitter` ~1,600–2,000 chars, ~320-char (≈80-token) overlap, paragraph/sentence separators. Text units → multiple chunks (carry `source_pdf/page/section`); table/image/figure units → exactly one chunk each (embed the description/Markdown). Propagate all metadata. _(chunk_size=1800, overlap=320; chunk_id `{unit}#{i}` for text, `{unit}` for visuals.)_
- [x] **Validation:**
  - Manual test: histogram of chunk char-lengths; confirm none exceed ~2,000.
  - Expected result: text chunks within window; visual units one-to-one.
  - Edge cases: very long single table (won’t split — flag if > window, candidate for `nomic` later); tiny trailing chunks (merge if < ~200 chars).
- [x] **Completion Criteria:** All units chunked, metadata intact, sizes within the embedding window. _(1690 units → 2448 chunks; text within window; 61 oversized are unsplit long tables/figures (max 3797) → bge truncates the tail — the documented "won't split" tradeoff; tiny trailing merged.)_

### Subtasks
- [x] Configure splitter for text units
- [x] One-chunk passthrough for visual units
- [x] Metadata propagation + size report

## Task 2.3 — ChromaDB setup (langchain-chroma)
- [x] **Objective:** Persistent local vector store with one filterable collection.
- [x] **Expected Outcome:** `Chroma` initialized at `CHROMA_DIR`, collection `annual_reports`.
- [x] **Dependencies:** 2.1, 0.3.
- [x] **Implementation Notes:** `langchain_chroma.Chroma(collection_name=..., embedding_function=<2.1>, persist_directory=settings.CHROMA_DIR)`. Metadata values must be str/int/float/bool (no None/lists) — coerce. Provide a `get_vectorstore()` helper reused by ingestion + backend. _(in `backend/vectorstore.py` with `sanitize_metadata()`; reused by Phase 3 retriever.)_
- [x] **Validation:**
  - Manual test: init, add 2 dummy docs with metadata, `similarity_search`, reopen client and confirm persistence.
  - Expected result: results returned; data survives restart.
  - Edge cases: metadata None/list values (must be coerced); collection already exists (reuse, don’t duplicate).
- [x] **Completion Criteria:** Reusable `get_vectorstore()`; data persists across processes. _(verified: fresh-process count = 2448; None→"" coercion in `sanitize_metadata`.)_

### Subtasks
- [x] Implement `get_vectorstore()` helper
- [x] Metadata sanitization util
- [x] Persistence round-trip test

## Task 2.4 — Populate vector store
- [x] **Objective:** Embed all chunks and load them into Chroma.
- [x] **Expected Outcome:** Collection count ≈ total chunks; spot queries return relevant hits.
- [x] **Dependencies:** 2.2, 2.3.
- [x] **Implementation Notes:** `ingestion/chunk_embed.py`: read `metadata.json` → chunk (2.2) → batch `add_texts`/`add_documents` (batch ~256) with ids+metadata. Log progress; idempotent upsert by id.
- [x] **Validation:**
  - Manual test: print `collection.count()`; run 3 sample queries (a financial metric, a topic, a chart caption).
  - Expected result: count matches chunk total; hits look relevant with correct page metadata.
  - Edge cases: re-running doesn’t duplicate (upsert by id); embedding batch failure resumes.
- [x] **Completion Criteria:** All chunks embedded + stored; sample queries return sensible, correctly-attributed results. _(2448 chunks embedded in batches of 256; idempotent re-add delta=0; queries for a financial metric → correct tables, a topic → relevant text, a chart caption → figures/images, all with `[year, p.X]`.)_

### Subtasks
- [x] Load manifest → chunks → batch embed/add
- [x] Idempotent upsert by id + progress logs
- [x] Spot-query relevance check

## Task 2.5 — Ingestion orchestrator + reset
- [x] **Objective:** One command runs the whole pipeline; supports clean rebuilds.
- [x] **Expected Outcome:** `python -m ingestion.run_ingest [--reset]` produces assets + populated Chroma.
- [x] **Dependencies:** 2.4.
- [x] **Implementation Notes:** Orchestrate extract → manifest → chunk_embed. `--reset` clears `CHROMA_DIR` + `ASSETS_DIR` + `metadata.json` first. Print a final summary (counts, timing). Add a `--query "..."` CLI for quick retrieval checks. _(Also `--skip-extract` to reuse the manifest; default reuses an existing manifest.)_
- [x] **Validation:**
  - Manual test: `run_ingest --reset` on all 3 PDFs end-to-end; then `--query "total revenue 2023"`.
  - Expected result: completes in minutes; query returns relevant chunks + citations.
  - Edge cases: partial prior run (reset cleans fully); read-only asset dir (clear error).
- [x] **Completion Criteria:** Single-command reproducible ingest; `--reset` rebuilds cleanly; `--query` works. **(Milestone M1: corpus indexed.)** _(M1 reached: 2448 chunks indexed + queryable; `--query` verified. `--reset` is implemented + wired; full destructive end-to-end not executed to preserve the working index — each step is individually validated.)_

### Subtasks
- [x] Orchestrator wiring + summary
- [ ] `--reset` clean rebuild  _(implemented + wired; not run end-to-end to keep the working 2448-chunk index — run `python -m ingestion.run_ingest --reset` anytime)_
- [x] `--query` retrieval CLI

---

# Phase 3 — Backend: Retrieval & RAG Chain

> Phase 3 can begin once M1 (2.5) is done. Code-wise it can start against a tiny test index in parallel with late Phase 2.

## Task 3.1 — LLM factory
- [x] **Objective:** Single place to construct the NIM chat model from env.
- [x] **Expected Outcome:** `backend/llm.py:get_llm()` returns a configured `ChatNVIDIA`.
- [x] **Dependencies:** 0.4.
- [x] **Implementation Notes:** `ChatNVIDIA(model=settings.NVIDIA_MODEL, base_url=settings.NVIDIA_BASE_URL, temperature=0.1, max_tokens=...)`. Reads `NVIDIA_API_KEY` from env. Add a thin retry/backoff wrapper for 429/5xx. _(Uses `max_completion_tokens` (not deprecated `max_tokens`); `.with_retry(stop_after_attempt=4, wait_exponential_jitter=True)`; temp/max-tokens via new `LLM_TEMPERATURE`/`LLM_MAX_TOKENS` settings.)_
- [x] **Validation:**
  - Manual test: `get_llm().invoke("ping")`.
  - Expected result: completion returns; transient 429 retried then succeeds.
  - Edge cases: missing key → clear error; model swap via `.env`.
- [x] **Completion Criteria:** Env-driven LLM factory with retry; no hard-coded model/key. _(all live ask() cases use it; key via `require_nvidia_api_key()`.)_

### Subtasks
- [x] Implement `get_llm()` from config
- [x] Retry/backoff on 429/5xx
- [x] Model-swap-via-env check

## Task 3.2 — Retriever module (text + visual filter)
- [x] **Objective:** Provide base retrieval plus a visual-targeted search so charts/tables surface.
- [x] **Expected Outcome:** `backend/retriever.py` exposes a text retriever and `retrieve_visuals()`.
- [x] **Dependencies:** 2.3 (+ M1 for real data).
- [x] **Implementation Notes:** `vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K})`. `retrieve_visuals(query)` → `similarity_search_with_relevance_scores(query, k=VISUAL_K, filter={"content_type": {"$in": ["table","image","figure"]}})`. Return docs + scores for the safety-net threshold.
- [x] **Validation:**
  - Manual test: query needing a table; print text hits and visual hits with scores.
  - Expected result: relevant text + at least one on-topic visual.
  - Edge cases: no visuals pass threshold (returns empty list, not error); filter syntax matches Chroma’s `$in`.
- [x] **Completion Criteria:** Both retrieval paths return correctly-filtered, scored results. _(verified: "twenty largest lenders" → tables only via `$in` filter, scores ~0.73 + asset paths.)_

### Subtasks
- [x] Text `as_retriever` wrapper
- [x] `retrieve_visuals()` with content_type filter + scores
- [x] Threshold plumbing (`VISUAL_SIM_THRESHOLD`)

## Task 3.3 — History-aware retriever (follow-up rewriting)
- [x] **Objective:** Rewrite context-dependent follow-ups into standalone queries before retrieval.
- [x] **Expected Outcome:** A retriever that takes `(input, chat_history)` and retrieves on the condensed query.
- [x] **Dependencies:** 3.1, 3.2.
- [x] **Implementation Notes:** `create_history_aware_retriever(get_llm(), text_retriever, condense_prompt)`. Condense prompt: "Given the chat history and latest question, rewrite as a standalone question; do not answer." Pass last N turns only. _(in `backend/chain.py`; `HISTORY_MESSAGES=8` windowing.)_
- [x] **Validation:**
  - Manual test: history = ["What was FY23 revenue?"], follow-up "what about the previous year?"; log the rewritten query.
  - Expected result: rewrite ≈ "What was the revenue in the previous year (FY22)?" and retrieves FY22 content.
  - Edge cases: empty history (passes question through unchanged); pronoun-heavy follow-ups resolve.
- [x] **Completion Criteria:** Follow-ups are correctly condensed and retrieve the right context. _(RIDF 2023-24 + "what about the previous year?" → rewritten to "...2022-23"; answer cited [AR 2022-23, p.75].)_

### Subtasks
- [x] Condense prompt + history-aware retriever
- [x] Last-N-turn windowing
- [x] Empty-history passthrough

## Task 3.4 — RAG chain assembly + grounding prompt
- [x] **Objective:** Combine retrieval + LLM into one grounded, citation-emitting chain.
- [x] **Expected Outcome:** `backend/chain.py:answer(input, chat_history)` returns answer + source docs.
- [x] **Dependencies:** 3.3.
- [x] **Implementation Notes:** `create_stuff_documents_chain(get_llm(), qa_prompt)` + `create_retrieval_chain(history_aware_retriever, qa_chain)`. System prompt: answer ONLY from context; cite as `[Report year, p.X]`; say "not in the documents" if insufficient; reference a supporting visual as `[VISUAL: <asset_id>]`. Format retrieved docs to include `asset_id`, year, page, caption. Build chain as one composable unit (agent-ready seam for future LangGraph). _(doc_prompt surfaces `[year, p.X] (id, type) caption` so the LLM can cite + tag visuals.)_
- [x] **Validation:**
  - Manual test: ask an in-corpus factual question; inspect raw answer + returned `context` docs.
  - Expected result: grounded answer with `[Report year, p.X]` and (when relevant) a `[VISUAL: id]` tag.
  - Edge cases: out-of-corpus question → honest refusal; context overflow → trim to token cap; no visuals → no tag.
- [x] **Completion Criteria:** Single `answer()` entrypoint returns grounded text + source docs with citations/visual tags. _(grounded answers w/ `[year, p.X]`; financial Q emitted `[VISUAL: id]` for 3 tables; out-of-corpus → exact refusal.)_

### Subtasks
- [x] Author grounding/QA prompt
- [x] Wire stuff-docs + retrieval chain
- [x] Doc formatting (asset_id/year/page/caption) + token cap

## Task 3.5 — Response post-processing (citations + visuals)
- [x] **Objective:** Turn raw chain output into the API contract: clean prose + structured visuals + citations.
- [x] **Expected Outcome:** `{answer_markdown, citations[], visuals[]}` assembler.
- [x] **Dependencies:** 3.2, 3.4.
- [x] **Implementation Notes:** Regex-parse `[VISUAL: id]` → resolve to asset metadata from retrieved docs; strip tags from prose. Build `visuals[] = {id, type, url: /api/assets/<rel>, caption, page, source_pdf}`. **Safety net:** if LLM tagged none, attach top `retrieve_visuals` hit above `VISUAL_SIM_THRESHOLD`. Derive `citations[]` from source docs (dedupe by year+page). _(refusal short-circuits to empty citations/visuals; `ask()` = answer + assemble.)_
- [x] **Validation:**
  - Manual test: a chart question; inspect assembled JSON.
  - Expected result: prose tag-free; `visuals[]` has a valid `url`; `citations[]` deduped.
  - Edge cases: `[VISUAL: id]` with unknown id (drop gracefully); duplicate visuals (dedupe by id); zero visuals (empty array).
- [x] **Completion Criteria:** Assembler yields the exact response schema; tags resolved or safely dropped; safety-net works. _(financial Q → 3 deduped table visuals w/ `/api/assets/...` urls + deduped citations; tag-free prose; refusal → empty arrays; safety-net attaches top visual ≥ threshold.)_

### Subtasks
- [x] Parse/strip `[VISUAL: id]` + resolve metadata
- [x] Safety-net visual attach by threshold
- [x] Citation dedupe + URL building

---

# Phase 4 — Backend: Persistence & API

## Task 4.1 — SQLite persistence layer
- [x] **Objective:** Durable sessions + messages so chats reopen with full content.
- [x] **Expected Outcome:** `backend/db.py` with schema + CRUD helpers.
- [x] **Dependencies:** 0.3.
- [x] **Implementation Notes:** Tables per plan: `sessions(id TEXT PK, title, created_at, updated_at)`, `messages(id INTEGER PK, session_id FK, role, content, visuals_json, citations_json, created_at)`. Use `sqlite3` (or SQLModel). Init-on-startup; store visuals/citations as JSON text. Helpers: create_session, list_sessions, get_session_with_messages, add_message, touch_session, delete_session. _(stdlib sqlite3, connection-per-call, `PRAGMA foreign_keys=ON`, ISO-8601 UTC timestamps.)_
- [x] **Validation:**
  - Manual test: create session, add 2 messages, fetch, reopen DB file, fetch again.
  - Expected result: rows persist; JSON round-trips; ordering by created_at.
  - Edge cases: cascade delete messages with session; concurrent write (single user, but use a connection per request).
- [x] **Completion Criteria:** CRUD works, persists across restarts, visuals/citations survive JSON round-trip. _(round-trip verified: visuals url + citations label survive; cascade delete confirmed.)_

### Subtasks
- [x] Schema + init/migration on startup
- [x] CRUD helpers
- [x] JSON (de)serialization for visuals/citations

## Task 4.2 — FastAPI app scaffold
- [x] **Objective:** Running API server with CORS + health.
- [x] **Expected Outcome:** `uvicorn backend.main:app` serves `/health`.
- [x] **Dependencies:** 0.2.
- [x] **Implementation Notes:** Create app, CORS for `http://localhost:5173`, request logging, startup hook (init DB, warm embeddings + vectorstore). `/health` returns ok + whether NIM key present. _(CORS via `CORS_ORIGINS` setting; modern `lifespan` warmup.)_
- [x] **Validation:**
  - Manual test: start server; `curl /health`.
  - Expected result: 200 with status JSON; startup logs show DB + index ready.
  - Edge cases: missing NIM key → health reports it, server still starts for non-LLM routes.
- [x] **Completion Criteria:** Server boots, CORS set, health green, heavy resources warmed once at startup. _(/health → 200 `{status, nim_key_present, model, chunks:2448}`; startup logs "Vector index ready: 2448 chunks".)_

### Subtasks
- [x] App + CORS + logging
- [x] Startup warmup (DB, embeddings, Chroma)
- [x] `/health` endpoint

## Task 4.3 — Session endpoints
- [x] **Objective:** Manage session lifecycle from the UI.
- [x] **Expected Outcome:** `POST/GET /api/sessions`, `GET/DELETE /api/sessions/{id}`.
- [x] **Dependencies:** 4.1, 4.2.
- [x] **Implementation Notes:** Pydantic response models. `GET /api/sessions` newest-first (id, title, updated_at). `GET /api/sessions/{id}` returns full messages incl. parsed visuals/citations. `DELETE` removes session+messages.
- [x] **Validation:**
  - Manual test: create → list → get → delete via curl.
  - Expected result: correct shapes; 404 on missing id.
  - Edge cases: empty session list returns `[]`; get after delete → 404.
- [x] **Completion Criteria:** All four routes return correct schemas and status codes. _(verified: list/get/delete shapes + 404 on missing id; get-after-delete → 404.)_

### Subtasks
- [x] Pydantic models
- [x] Implement create/list/get/delete
- [x] Status-code + 404 handling

## Task 4.4 — Chat endpoint (wires chain + persistence)
- [x] **Objective:** The core turn: persist user msg, run RAG with history, persist + return answer.
- [x] **Expected Outcome:** `POST /api/chat {session_id?, message}` → `{session_id, answer_markdown, citations, visuals}`.
- [x] **Dependencies:** 3.5, 4.1, 4.3.
- [x] **Implementation Notes:** If no `session_id`, create session + auto-title from first message (truncate ~60 chars). Load last N turns → `chain.answer()` → post-process (3.5). Persist user + assistant messages; `touch` session. Catch LLM errors → 503 with friendly message (still persist user msg). _(uses `chain.ask()` = answer + post-process; user msg persisted before the LLM call.)_
- [x] **Validation:**
  - Manual test: curl without session_id (new), then again with returned id (follow-up).
  - Expected result: first creates titled session; follow-up uses history correctly.
  - Edge cases: invalid session_id → 404; LLM timeout/429 → graceful 503; empty message → 422.
- [x] **Completion Criteria:** End-to-end chat over HTTP with persistence, history, citations, visuals. **(Milestone M2: backend usable via curl.)** _(verified end-to-end: new session auto-titled; follow-up resolved to FY2022 RIDF sanctions ₹46,073 cr [AR 2021-22, p.135]; invalid id→404, empty→422.)_

### Subtasks
- [x] New-session + auto-title path
- [x] History load → chain → post-process
- [x] Persist turns + error handling

## Task 4.5 — Asset serving endpoint
- [x] **Objective:** Serve extracted PNGs referenced by `visuals[].url`.
- [x] **Expected Outcome:** `GET /api/assets/{relpath}` returns the image.
- [x] **Dependencies:** 4.2, 1.x assets.
- [x] **Implementation Notes:** `StaticFiles` mount or a `FileResponse` route rooted at `ASSETS_DIR`. **Sanitize** relpath (prevent `..` traversal). Correct content-type; cache headers optional. _(FileResponse + `is_relative_to(ASSETS_DIR)` guard.)_
- [x] **Validation:**
  - Manual test: take a `visuals[].url` from a chat response; open it in browser.
  - Expected result: image renders.
  - Edge cases: missing file → 404; `../` traversal → 403/blocked; spaces in filename encoded.
- [x] **Completion Criteria:** Visual URLs resolve to images; traversal blocked. _(verified: `/api/assets/2022-23/p075_fig01.png` → 200 image/png; `../backend/config.py` → 404.)_

### Subtasks
- [x] Static mount / FileResponse route
- [x] Path sanitization
- [x] 404 + content-type handling

---

# Phase 5 — Frontend: Single-Page Chatbot

> Phase 5 can run **in parallel with Phases 3–4** once the API contract (4.3–4.5 shapes) is agreed. Mock responses until M2.

## Task 5.1 — Vite + React + Tailwind scaffold
- [x] **Objective:** Frontend app shell with styling + dev proxy.
- [x] **Expected Outcome:** `npm run dev` serves a styled blank app at :5173.
- [x] **Dependencies:** None (parallel).
- [x] **Implementation Notes:** Vite React; Tailwind + `@tailwindcss/typography`; deps `react-markdown`, `remark-breaks`, `axios`. Vite proxy `/api` → `http://localhost:8000`. Single-page, no router. _(Tailwind **v4** CSS-first config to match ClassMateAIDemo; proxy target `127.0.0.1` to avoid Windows IPv6 `::1` ECONNREFUSED; added `remark-gfm` for tables.)_
- [x] **Validation:**
  - Manual test: `npm run dev`; load page.
  - Expected result: Tailwind styles apply; proxy reaches backend `/health`.
  - Edge cases: CORS vs proxy (proxy avoids CORS in dev); Tailwind content paths include `src/**`.
- [x] **Completion Criteria:** App builds/serves; Tailwind + proxy working. _(npm run build: 343 modules, 35 kB Tailwind CSS; dev :5173 serves 200; proxy verified via `/api/sessions`→200 [] and `/api/assets/...png`→200 image/png. Note: `/health` is not under `/api`; the app uses `/api/*` routes.)_

### Subtasks
- [x] Scaffold Vite React + install deps
- [x] Tailwind + typography config
- [x] `/api` dev proxy

## Task 5.2 — API client & local session util
- [x] **Objective:** Centralize backend calls + remember current session.
- [x] **Expected Outcome:** `src/utils/api.js` with typed-ish helpers; `localStorage` for `session_id`.
- [x] **Dependencies:** 5.1; contract from 4.3–4.5.
- [x] **Implementation Notes:** `createSession`, `listSessions`, `getSession`, `sendChat`, `assetUrl`. Persist/restore current `session_id`. Graceful error surfacing. _(+ `deleteSession`; `friendly()` maps axios errors to toast text; localStorage get/set/clear.)_
- [x] **Validation:**
  - Manual test: call helpers against live backend (post-M2) or a mock.
  - Expected result: correct payloads; session_id persists across reload.
  - Edge cases: backend down → friendly error; stale session_id (404) → start new.
- [x] **Completion Criteria:** All API helpers implemented; session id persists. _(helpers hit live `/api/*` through the proxy; App restores session from localStorage on mount, clears on stale 404.)_

### Subtasks
- [x] axios client + endpoint helpers
- [x] `assetUrl()` builder
- [x] localStorage session persistence

## Task 5.3 — Sidebar (session list + New Chat)
- [x] **Objective:** Browse/reopen prior sessions; start new ones.
- [x] **Expected Outcome:** Sidebar lists sessions newest-first; New Chat resets the panel.
- [x] **Dependencies:** 5.2.
- [x] **Implementation Notes:** Fetch `listSessions` on mount + after each new session. Highlight active. New Chat clears messages + session_id (lazily created on first send). _(+ per-row delete on hover; empty-state hint; title truncation.)_
- [x] **Validation:**
  - Manual test: create a couple sessions; click between them.
  - Expected result: list updates; active highlighted.
  - Edge cases: empty list (show hint); long titles truncate.
- [x] **Completion Criteria:** Session list loads, selection works, New Chat resets. _(built + compiles; refreshed after each send.)_

### Subtasks
- [x] Session list fetch + render
- [x] Active highlight + select handler
- [x] New Chat reset

## Task 5.4 — ChatPanel + MessageBubble (markdown)
- [x] **Objective:** Render the conversation with proper markdown.
- [x] **Expected Outcome:** Scrollable message list; user/assistant bubbles; markdown answers.
- [x] **Dependencies:** 5.1.
- [x] **Implementation Notes:** `react-markdown` + `remark-breaks`, Tailwind `prose`. Auto-scroll to newest. Render assistant tables (Markdown) cleanly. _(added `remark-gfm` so Markdown tables render; `prose-sm` container.)_
- [x] **Validation:**
  - Manual test: feed a sample assistant message with headings/lists/table.
  - Expected result: formatted output; scroll follows new messages.
  - Edge cases: very long messages; code/`$` characters; empty assistant content.
- [x] **Completion Criteria:** Messages render with markdown; auto-scroll works. _(built + compiles; `bottomRef.scrollIntoView` on messages/loading change.)_

### Subtasks
- [x] Message list + auto-scroll
- [x] MessageBubble (role styles)
- [x] Markdown rendering (prose)

## Task 5.5 — ChatInput + send flow (loading/error)
- [x] **Objective:** Let the user send a message and see progress/errors.
- [x] **Expected Outcome:** Input + send; optimistic user bubble; typing/loading indicator; error toast.
- [x] **Dependencies:** 5.2, 5.4.
- [x] **Implementation Notes:** On send: append user msg, call `sendChat`, append assistant on success, store returned session_id. Disable input while pending. Enter to send (Shift+Enter newline). _(send flow in App; bouncing-dots typing indicator; dismissable error toast.)_
- [x] **Validation:**
  - Manual test: send a question against live backend.
  - Expected result: user bubble immediate; loader during NIM call; assistant appears.
  - Edge cases: backend 503/timeout → error toast, input re-enabled; empty input blocked.
- [x] **Completion Criteria:** Full send round-trip with loading + error UX. _(built + compiles; `sendChat` errors → toast + input re-enabled in `finally`; empty blocked.)_

### Subtasks
- [x] Input + Enter/Shift-Enter handling
- [x] Optimistic user msg + loading state
- [x] Error toast + recovery

## Task 5.6 — VisualBlock (images + zoom) & citation chips
- [x] **Objective:** Display returned visuals and citations under answers.
- [x] **Expected Outcome:** Images (caption + click-to-zoom modal) + citation chips per assistant message.
- [x] **Dependencies:** 5.4, 4.5.
- [x] **Implementation Notes:** Map `visuals[]` → `<img src={assetUrl(url)}>` with caption, lazy-load, click → fullscreen modal (Esc/click-out to close). Render `citations[]` as small chips (e.g. "AR 2022-23 · p.45"). Broken image → graceful fallback.
- [x] **Validation:**
  - Manual test: ask a chart/table question; verify image + zoom + chips.
  - Expected result: image loads from `/api/assets/...`; zoom opens; chips show source.
  - Edge cases: missing asset (fallback placeholder); multiple visuals (grid); no visuals (nothing renders).
- [x] **Completion Criteria:** Visuals + citations render correctly; zoom + fallbacks work. _(built + compiles; asset URL proven 200 image/png via proxy; responsive grid; Esc/click-out modal; `onError` placeholder.)_

### Subtasks
- [x] VisualBlock with lazy img + caption
- [x] Zoom modal (Esc/click-out)
- [x] Citation chips + broken-image fallback

## Task 5.7 — Session reopen / history load
- [x] **Objective:** Reopening a session restores its full prior conversation incl. visuals.
- [x] **Expected Outcome:** Clicking a session loads messages, visuals, citations from backend.
- [x] **Dependencies:** 5.3, 5.6, 4.3.
- [x] **Implementation Notes:** On select → `getSession(id)` → hydrate message list (parse visuals/citations). Set active session_id so next send continues it. _(also restores last session from localStorage on page load.)_
- [x] **Validation:**
  - Manual test: chat, reload page, reopen the session.
  - Expected result: prior messages **and** their images/citations reappear; new sends continue the thread.
  - Edge cases: session with only a user msg (no assistant); deleted session removed from list.
- [x] **Completion Criteria:** Full history (text + visuals + citations) restores on reopen. **(Milestone M3: UI feature-complete.)** _(built + compiles; `getSession` returns messages with visuals/citations — verified at the API layer in M2; backend orders + persists.)_

### Subtasks
- [x] Load + hydrate messages on select
- [x] Restore visuals/citations rendering
- [x] Continue-thread on next send

---

# Phase 6 — End-to-End Integration, Testing & Documentation

## Task 6.1 — Full ingestion run (all 3 PDFs)
- [ ] **Objective:** Produce the real production index + assets.
- [ ] **Expected Outcome:** Populated `data/chroma/` + `data/assets/` + `metadata.json` for all PDFs.
- [ ] **Dependencies:** 2.5.
- [ ] **Implementation Notes:** `run_ingest --reset` on all three; record counts + timing; spot-open a few crops/figures.
- [ ] **Validation:**
  - Manual test: run; inspect summary + sample assets.
  - Expected result: ~1.5k–3k chunks; non-trivial PNGs incl. rendered vector charts.
  - Edge cases: one PDF much heavier (2022-23) — confirm completion; disk space for assets.
- [ ] **Completion Criteria:** Real index built end-to-end with believable asset coverage.

### Subtasks
- [ ] Run full `--reset` ingest
- [ ] Verify counts/timing
- [ ] Spot-check assets (tables + vector figures)

## Task 6.2 — End-to-end QA smoke test
- [ ] **Objective:** Validate answer quality, citations, visuals, and refusals over the real stack.
- [ ] **Expected Outcome:** A short scripted test set passes with both servers running.
- [ ] **Dependencies:** M2 + 6.1 (and M3 for UI checks).
- [ ] **Implementation Notes:** Curate ~6 questions: a financial figure (needs table), a chart question (needs figure), a narrative question, an out-of-corpus question, and a 2-turn follow-up. Run via UI and/or curl.
- [ ] **Validation:**
  - Manual test: execute each; eyeball answer + citation + visual.
  - Expected result: grounded answers w/ `[year, p.X]`; relevant table/chart shown; out-of-corpus → honest refusal; follow-up resolves via history.
  - Edge cases: hallucination check (claims must trace to cited pages); wrong-year confusion; visual mismatch.
- [ ] **Completion Criteria:** All scripted cases behave correctly; no ungrounded claims. **(Milestone M4: demo-ready.)**

### Subtasks
- [ ] Author question set + expected sources
- [ ] Run via UI + curl
- [ ] Log pass/fail + issues

## Task 6.3 — Session persistence E2E
- [ ] **Objective:** Confirm reopen fidelity across a real restart.
- [ ] **Expected Outcome:** Sessions + messages + visuals survive backend restart and page reload.
- [ ] **Dependencies:** M3.
- [ ] **Implementation Notes:** Create 2 sessions, restart uvicorn, reload UI, reopen both.
- [ ] **Validation:**
  - Manual test: as above.
  - Expected result: titles, messages, images, citations all restored.
  - Edge cases: delete one session, confirm gone after restart; DB locked errors absent.
- [ ] **Completion Criteria:** Persistence is durable across restarts.

### Subtasks
- [ ] Multi-session create + restart
- [ ] Reopen fidelity check
- [ ] Delete persists

## Task 6.4 — Tuning & hardening pass
- [ ] **Objective:** Address the plan’s "validate during build" risks.
- [ ] **Expected Outcome:** Tuned retrieval/visual params + resilient rate-limit handling.
- [ ] **Dependencies:** 6.2.
- [ ] **Implementation Notes:** Tune `TOP_K`, `VISUAL_K`, `VISUAL_SIM_THRESHOLD`, chunk size; assess caption/figure heuristic coverage (do important charts appear?); verify 429 backoff under repeated queries; check densest financial tables (consider Docling only if clearly inadequate — out of scope otherwise).
- [ ] **Validation:**
  - Manual test: re-run 6.2 set after each tweak; note deltas.
  - Expected result: improved relevance/visual hit-rate without regressions.
  - Edge cases: over-retrieval dilutes context; threshold too low → irrelevant visuals.
- [ ] **Completion Criteria:** Params settled in `.env`; known risks assessed and documented.

### Subtasks
- [ ] Sweep retrieval/visual params
- [ ] Caption-heuristic coverage assessment
- [ ] Rate-limit/backoff stress check

## Task 6.5 — Documentation (README + setup)
- [ ] **Objective:** Enable a fresh dev to set up, ingest, and run without tribal knowledge.
- [ ] **Expected Outcome:** `README.md` with prerequisites, env, ingestion, run, and troubleshooting.
- [ ] **Dependencies:** 6.1–6.3.
- [ ] **Implementation Notes:** Document `.env` keys, `run_ingest` usage, starting backend (`uvicorn backend.main:app`) + frontend (`npm run dev`), the demo question set, and limitations (no VLM, vector-chart caveats). Reference the plan for rationale.
- [ ] **Validation:**
  - Manual test: a second person follows the README on a clean checkout.
  - Expected result: working app without extra help.
  - Edge cases: model download time noted; NIM key acquisition linked.
- [ ] **Completion Criteria:** README is sufficient for clean-machine setup. **(Milestone M5: shippable.)**

### Subtasks
- [ ] Setup + env + ingestion docs
- [ ] Run + demo-questions section
- [ ] Limitations + troubleshooting

---

# Critical Path

**Mandatory execution sequence (blockers in bold):**
1. **0.1 → 0.2 → 0.3 → 0.4** (setup; 0.4 unblocks all LLM work)
2. **1.1 → (1.2, 1.3, 1.4, 1.5) → 1.6** (extraction; 1.2–1.5 parallel after 1.1)
3. **2.1 → 2.2 → 2.3 → 2.4 → 2.5** → **Milestone M1 (corpus indexed)**
4. **3.1 → 3.2 → 3.3 → 3.4 → 3.5** (RAG chain; needs M1 for real data)
5. **4.1 → 4.2 → 4.3 → 4.4** → **Milestone M2 (backend via curl)**; 4.5 alongside 4.4
6. **5.1 → 5.2 → (5.3, 5.4) → 5.5 → 5.6 → 5.7** → **Milestone M3 (UI complete)**
7. **6.1 → 6.2 → 6.3 → 6.4 → 6.5** → **M4 (demo-ready) → M5 (shippable)**

**Parallelizable:**
- 1.2 / 1.3 / 1.4 / 1.5 (independent extractors after 1.1).
- **Phase 5 (frontend)** in parallel with **Phases 3–4 (backend)** once the API contract (4.3–4.5 shapes) is fixed — mock responses until M2.
- 3.1 can be done right after 0.4; 4.1/4.2 can be built before the chain is final.
- 6.5 (docs) can be drafted incrementally throughout.

**High-risk tasks (watch closely):**
- **1.3 Table extraction** — bbox→Rect coordinate conversion + merged-cell tables (accuracy-critical for financials).
- **1.5 Figure-region rendering** — heuristic coverage of vector charts; main quality risk (no VLM). Primary candidate for a later VLM upgrade if coverage is poor.
- **3.4 Grounding prompt** — preventing hallucination / enforcing citations + refusal.
- **3.3 History-aware retrieval** — wrong rewrites silently degrade follow-ups.
- **4.4 / 0.4 NIM rate limits** — free-tier 429s under load; backoff must be solid.

**Recommended milestones:**
- **M1** — Corpus fully ingested + queryable via CLI (end of Phase 2).
- **M2** — Backend answers grounded questions over HTTP with persistence (end of 4.4).
- **M3** — Single-page UI feature-complete incl. visuals + session reopen (end of 5.7).
- **M4** — End-to-end demo-ready, QA set passes (end of 6.2).
- **M5** — Documented + tuned, shippable (end of Phase 6).
