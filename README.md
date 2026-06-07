# RAG Chatbot over Annual Reports

A demo Retrieval-Augmented-Generation chatbot that answers questions **strictly from a
set of annual-report PDFs** and, when relevant, shows the supporting **tables, charts,
and images** alongside the grounded text answer (with page-level citations). Chat
sessions persist locally so you can reopen a prior conversation with its full content.

- **Generation LLM:** NVIDIA NIM cloud API via LangChain `ChatNVIDIA` (model/key/base-URL in `.env`)
- **Embeddings:** local `BAAI/bge-small-en-v1.5` (384-dim, offline, free)
- **Vector store:** ChromaDB (local, persistent) via `langchain-chroma`
- **Backend:** FastAPI + SQLite · **Frontend:** React 19 + Vite + Tailwind v4 (single page)
- **Orchestration:** LangChain (history-aware retrieval chain; agent-ready seam for later LangGraph)

> Architecture & rationale: [`Implementation plan/IMPLEMENTATION_PLAN.md`](Implementation%20plan/IMPLEMENTATION_PLAN.md).
> Task-by-task tracker: [`Implementation plan/TASK_LIST.md`](Implementation%20plan/TASK_LIST.md).

---

## Features

- Grounded answers only from the PDFs; honest *"I could not find this in the documents."* when unknown.
- Inline citations as `[Report year, p.X]`; deduped citation chips under each answer.
- Returned **visuals** (financial tables as cropped PNGs, embedded photos, and caption-anchored vector-chart renders) with click-to-zoom.
- Context-aware **follow-ups** (history rewrites "what about the previous year?" into a standalone query).
- Persistent, reopenable sessions (SQLite); auto-generated titles.

## Project structure

```
RAG System/
├── ingestion/          # offline pipeline
│   ├── extract.py        # PDFs -> text/tables/images/figures + metadata.json + asset PNGs
│   ├── chunk_embed.py    # chunk + bge-embed -> ChromaDB
│   └── run_ingest.py     # orchestrator: --reset / --query / --skip-extract
├── backend/            # online service
│   ├── config.py         # central, env-driven settings (single source of truth)
│   ├── embeddings.py     # shared bge embedder (ingest + query)
│   ├── vectorstore.py    # get_vectorstore() + metadata sanitization
│   ├── llm.py            # ChatNVIDIA factory (+ retry/backoff)
│   ├── retriever.py      # text retriever + visual-filtered search
│   ├── chain.py          # history-aware RAG chain + response post-processing (ask())
│   ├── db.py             # SQLite sessions/messages
│   └── main.py           # FastAPI app: /health, /api/sessions, /api/chat, /api/assets
├── frontend/           # React + Vite + Tailwind single page
│   └── src/{App.jsx, utils/api.js, components/{Sidebar,ChatPanel,MessageBubble,VisualBlock,ChatInput}.jsx}
├── pdf_for_RAG/        # source PDFs
├── data/               # chroma/  assets/  metadata.json  app.db   (git-ignored contents)
├── nim_smoke_test.py   # NVIDIA connectivity check
├── qa_smoke.py         # end-to-end QA harness
├── requirements.txt
└── .env.example
```

## Prerequisites

- **Python 3.11–3.13** (developed on 3.13)
- **Node.js 18+** and npm (for the frontend)
- A free **NVIDIA API key** from <https://build.nvidia.com>

---

## Setup

```powershell
# 1. Virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows PowerShell
# source .venv/bin/activate             # macOS / Linux

# 2. Python dependencies (first run downloads the ~130 MB bge model on first use)
pip install -r requirements.txt

# 3. Environment config
copy .env.example .env                  # cp .env.example .env  (macOS/Linux)
#   edit .env and set NVIDIA_API_KEY=nvapi-xxxx

# 4. Verify config + NVIDIA connectivity
python -c "from backend.config import settings; print(settings)"
python nim_smoke_test.py
```

### Configuration (`.env`)

Every tunable is environment-driven (relative paths resolve to the project root). Key settings:

| Key | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | — | NIM key (required for chat/smoke test) |
| `NVIDIA_MODEL` | `meta/llama-3.3-70b-instruct` | chat model (swap with no code change) |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible NIM endpoint |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | `0.1` / `1024` | generation sampling |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | local embedder (ingest + query) |
| `PDF_DIR` | `pdf_for_RAG` | source PDFs |
| `CHROMA_DIR` / `ASSETS_DIR` / `METADATA_PATH` / `SQLITE_PATH` | under `data/` | storage paths |
| `COLLECTION_NAME` | `annual_reports` | Chroma collection |
| `TOP_K` / `VISUAL_K` / `VISUAL_SIM_THRESHOLD` | `8` / `3` / `0.4` | retrieval + visual safety-net tuning |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | allowed frontend origins |

---

## Ingestion (build the index)

Run once (or whenever the PDFs change). Local embeddings → no API cost.

```powershell
python -m ingestion.run_ingest --reset           # full clean rebuild (~20+ min)
python -m ingestion.run_ingest --skip-extract     # re-embed from existing metadata.json
python -m ingestion.run_ingest --query "total deposits and borrowings"   # quick retrieval check
```

Produces `data/metadata.json` (~1.7k extraction units), `data/assets/<year>/*.png` (~900 crops/figures/images),
and a populated `data/chroma/` collection (~2.4k chunks).

## Run the app

```powershell
# Terminal 1 — backend  (http://127.0.0.1:8000, docs at /docs)
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload

# Terminal 2 — frontend (http://localhost:5173)
cd frontend
npm install        # first time only
npm run dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api/*` to the backend (no CORS in dev).

## Demo questions

- **Financial (table):** "What were the total borrowings from the twenty largest lenders?"
- **Chart/figure:** "Show the performance of the Rural Infrastructure Development Fund over the years."
- **Narrative:** "What is NABARD's role in rural development?"
- **Out-of-corpus (refusal):** "What is the population of Brazil?"
- **Follow-up:** "What was the RIDF allocation in 2023-24?" → then "what about the previous year?"

Quick automated check (backend running): `python qa_smoke.py`.

---

## How it works

**Offline:** each PDF is parsed page-by-page (PyMuPDF) into text (cleaned, header/footer-stripped,
section-headed), ruled tables (Markdown + a cropped PNG), embedded images (RGB, logo-filtered), and
caption-anchored vector-chart region renders → a `metadata.json` manifest + asset PNGs. Units are
chunked and embedded with bge into ChromaDB.

**Online:** a follow-up is rewritten to a standalone query (history-aware retriever) → top-k retrieval
(+ a visual-filtered search) → a grounding prompt instructs the NIM model to answer only from context,
cite `[year, p.X]`, and tag supporting visuals as `[VISUAL: id]` → post-processing yields
`{answer_markdown, citations[], visuals[]}`, persisted to SQLite and rendered by the UI.

## Limitations

- **No vision model.** Vector-drawn charts are surfaced via **caption-anchored region rendering**, so some
  charts may be missed or imperfectly cropped — factual accuracy is carried by text + extractable tables.
- **PDF parsing** uses **PyMuPDF** for text/tables (pdfplumber was ~100× slower on these graphics-heavy
  reports); complex/merged-cell tables can be imperfect.
- A few very long single tables/figures exceed the bge 512-token window and are truncated at embedding time.
- **NIM free tier** adds latency and occasional 429s (handled by retry/backoff). `bge-small` + no reranker
  set a retrieval-quality ceiling.
- **LangChain is pinned to 0.3.x** (the documented chain APIs); do not bump to v1.x without porting.
- Single local user; no auth/concurrency.

## Troubleshooting

- **`NVIDIA_API_KEY is not set`** → add it to `.env`; get one at build.nvidia.com.
- **First query is slow** → the bge model (~130 MB) downloads/loads once; the backend warms it at startup.
- **Frontend can't reach the API (`ECONNREFUSED ::1:8000`)** → the Vite proxy targets `127.0.0.1:8000`;
  ensure the backend is running and bound to `127.0.0.1`.
- **Port already in use** → stop the other process or change the port (`uvicorn ... --port 8001`, and the
  Vite proxy target accordingly).
- **Windows console can't print `₹`** → set `PYTHONIOENCODING=utf-8` for CLI scripts (data is stored UTF-8).
- **Model unavailable / 404** → change `NVIDIA_MODEL` in `.env` and re-run (no code change).

## Tech stack

Python · LangChain 0.3 · langchain-nvidia-ai-endpoints (`ChatNVIDIA`) · sentence-transformers (bge) ·
ChromaDB · PyMuPDF · FastAPI · SQLite · React 19 · Vite · Tailwind v4 · react-markdown.
