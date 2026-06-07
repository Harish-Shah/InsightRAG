"""FastAPI application (Tasks 4.2-4.5) - Milestone M2.

Wraps the RAG chain (`chain.ask`) and SQLite persistence (`db`) in an HTTP API:

    GET  /health
    POST /api/sessions          GET /api/sessions
    GET  /api/sessions/{id}      DELETE /api/sessions/{id}
    POST /api/chat              -> grounded answer + citations + visuals (persisted)
    GET  /api/assets/{relpath}  -> serves extracted PNGs (traversal-safe)

Run:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend import db
from backend.chain import ask
from backend.config import settings
from backend.embeddings import get_embeddings
from backend.vectorstore import collection_count, get_vectorstore

logger = logging.getLogger("rag.api")


# --- Lifespan: init DB + warm heavy resources once --------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings.ensure_dirs()
    db.init_db()
    logger.info("DB ready at %s", settings.SQLITE_PATH)
    try:
        get_embeddings()                       # load bge model into RAM
        n = get_vectorstore()._collection.count()
        logger.info("Vector index ready: %s chunks", n)
    except Exception as exc:  # noqa: BLE001 - server should still start
        logger.warning("Index warmup failed (queries may be slow/empty): %s", exc)
    logger.info("NIM key present: %s | model: %s",
                settings.has_nvidia_key, settings.NVIDIA_MODEL)
    yield


app = FastAPI(title="RAG Chatbot API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    dur_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.0fms)",
                request.method, request.url.path, response.status_code, dur_ms)
    return response


# --- Pydantic models ---------------------------------------------------------
class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class SessionSummary(BaseModel):
    id: str
    title: str
    updated_at: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    visuals: list[dict] = []
    citations: list[dict] = []
    created_at: str


class SessionDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageOut] = []


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    answer_markdown: str
    citations: list[dict] = []
    visuals: list[dict] = []


# --- Health ------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    try:
        chunks = collection_count()
    except Exception:  # noqa: BLE001
        chunks = None
    return {
        "status": "ok",
        "nim_key_present": settings.has_nvidia_key,
        "model": settings.NVIDIA_MODEL,
        "chunks": chunks,
    }


# --- Sessions ----------------------------------------------------------------
@app.post("/api/sessions", response_model=SessionSummary)
def create_session(req: CreateSessionRequest) -> SessionSummary:
    title = (req.title or "").strip() or "New Chat"
    sid = db.create_session(title)
    s = db.get_session(sid)
    return SessionSummary(id=s["id"], title=s["title"], updated_at=s["updated_at"])


@app.get("/api/sessions", response_model=list[SessionSummary])
def list_sessions() -> list[SessionSummary]:
    return [
        SessionSummary(id=s["id"], title=s["title"], updated_at=s["updated_at"])
        for s in db.list_sessions()
    ]


@app.get("/api/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str) -> dict[str, Any]:
    session = db.get_session_with_messages(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    if not db.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


# --- Chat (wires chain + persistence) ---------------------------------------
def _auto_title(message: str) -> str:
    t = " ".join(message.strip().split())
    if not t:
        return "New Chat"
    return t[:57] + "..." if len(t) > 60 else t


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")

    # Resolve or create the session.
    session_id = req.session_id
    if session_id:
        if db.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session_id = db.create_session(_auto_title(message))

    # History = messages before this turn (for follow-up rewriting).
    detail = db.get_session_with_messages(session_id)
    history = [{"role": m["role"], "content": m["content"]} for m in detail["messages"]]

    # Persist the user message first so it survives an LLM failure.
    db.add_message(session_id, "user", message)

    try:
        result = ask(message, history)
    except Exception as exc:  # noqa: BLE001 - turn the failure into a friendly 503
        logger.exception("chat generation failed")
        db.touch_session(session_id)
        raise HTTPException(
            status_code=503,
            detail=f"The assistant is temporarily unavailable. ({exc})",
        )

    db.add_message(session_id, "assistant", result["answer_markdown"],
                   visuals=result["visuals"], citations=result["citations"])
    db.touch_session(session_id)

    return ChatResponse(
        session_id=session_id,
        answer_markdown=result["answer_markdown"],
        citations=result["citations"],
        visuals=result["visuals"],
    )


# --- Asset serving (traversal-safe) -----------------------------------------
@app.get("/api/assets/{relpath:path}")
def get_asset(relpath: str) -> FileResponse:
    base = settings.ASSETS_DIR.resolve()
    target = (base / relpath).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(target)
