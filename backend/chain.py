"""RAG chain assembly + response post-processing.

Composes the online query brain as ONE unit (agent-ready seam for a future
LangGraph tool):

    history-aware retriever (rewrites follow-ups)  ->  Chroma retrieval
        ->  stuff-documents chain (grounding prompt)  ->  ChatNVIDIA

and post-processes the raw output into the API contract::

    {answer_markdown, citations[], visuals[]}

Public entry point used by /api/chat:  ``ask(query, chat_history)``.
"""

from __future__ import annotations

import logging
import re
import time
from functools import lru_cache
from typing import Any, AsyncIterator, Optional

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from backend.config import settings
from backend.llm import get_llm
from backend.retriever import get_text_retriever, retrieve_visuals

log = logging.getLogger("rag.chain")

# Keep the last N messages of history in context (≈ last 4 turns).
HISTORY_MESSAGES = 8
REFUSAL = "I could not find this in the documents."
_VISUAL_TAG_RE = re.compile(r"\[VISUAL:\s*([^\]]+)\]", re.IGNORECASE)
_VISUAL_OPEN = "[VISUAL:"  # prefix we must never let flash mid-stream

# --- Prompts -----------------------------------------------------------------
_CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question (which may refer back "
     "to the conversation), rewrite it as a standalone question understandable "
     "without the chat history. Do NOT answer it - only reformulate if needed, "
     "otherwise return it unchanged."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

_QA_SYSTEM = (
    "You are a precise assistant answering questions about NABARD annual reports. "
    "Answer ONLY from the provided context - do not use outside knowledge or guess.\n"
    "Rules:\n"
    "1. Ground every claim in the context.\n"
    "2. Cite the source inline right after each fact as [Report year, p.X] using "
    "the report year and page shown for that context item "
    "(e.g. [Annual Report 2022-23, p.45]).\n"
    f"3. If the context does not contain the answer, reply exactly: \"{REFUSAL}\" "
    "and nothing else.\n"
    "4. When a context item of type table/figure/image supports your answer, add "
    "[VISUAL: <id>] using that item's exact id (shown as 'id: ...').\n"
    "5. Be concise; use Markdown, and render tabular answers as Markdown tables.\n\n"
    "Context:\n{context}"
)
_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _QA_SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# How each retrieved doc is rendered into {context} for the LLM (with provenance).
_DOC_PROMPT = PromptTemplate.from_template(
    "[{report_year}, p.{page}] (id: {chunk_id}, type: {content_type}) {caption}\n{page_content}"
)


# --- Chain construction ------------------------------------------------------
@lru_cache(maxsize=1)
def build_chain():
    """build the composable history-aware retrieval+QA chain (singleton)."""
    llm = get_llm()
    history_aware_retriever = create_history_aware_retriever(
        llm, get_text_retriever(), _CONDENSE_PROMPT
    )
    qa_chain = create_stuff_documents_chain(
        llm, _QA_PROMPT, document_prompt=_DOC_PROMPT, document_separator="\n\n---\n\n"
    )
    return create_retrieval_chain(history_aware_retriever, qa_chain)


def _to_messages(history: Optional[list]) -> list[BaseMessage]:
    """Normalize chat history (BaseMessage | dict | (role, content)) -> messages."""
    if not history:
        return []
    out: list[BaseMessage] = []
    for item in history:
        if isinstance(item, BaseMessage):
            out.append(item)
            continue
        if isinstance(item, dict):
            role, content = item.get("role", ""), item.get("content", "")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            role, content = item
        else:
            continue
        role = str(role).lower()
        if role in ("user", "human"):
            out.append(HumanMessage(content=content))
        elif role in ("assistant", "ai", "bot"):
            out.append(AIMessage(content=content))
    return out[-HISTORY_MESSAGES:]


def answer(query: str, chat_history: Optional[list] = None) -> dict[str, Any]:
    """Run the RAG chain. Returns {"answer": str, "context": list[Document]}."""
    out = build_chain().invoke({
        "input": query,
        "chat_history": _to_messages(chat_history),
    })
    return {"answer": out["answer"], "context": out.get("context", [])}


# --- Incremental [VISUAL: id] stripping (for live token streaming) ----------
def _safe_split(buf: str) -> tuple[str, str]:
    """Split a buffer into (emit_now, hold_back).

    Holds back only a trailing suffix that could still grow into a ``[VISUAL: ..]``
    tag, so the tag never flashes mid-stream. Inline citation brackets such as
    ``[Annual Report 2022-23, p.45]`` are NOT held back.
    """
    idx = buf.rfind("[")
    while idx != -1:
        tail = buf[idx:]
        upper = tail.upper()
        if upper.startswith(_VISUAL_OPEN):
            # An opened VISUAL tag: hold it back until its closing ']' arrives
            # (complete tags are already removed by the caller's regex sub).
            return buf[:idx], tail
        if _VISUAL_OPEN.startswith(upper):
            # A partial prefix of "[VISUAL:" (e.g. "[VI") -> hold back.
            return buf[:idx], tail
        # Some other bracket (e.g. a citation): check any earlier '['.
        idx = buf.rfind("[", 0, idx)
    return buf, ""


def _make_visual_stripper():
    """Stateful stripper: feed tokens, get back tag-free text safe to emit now."""
    buf = ""

    def feed(token: str) -> str:
        nonlocal buf
        buf += token
        buf = _VISUAL_TAG_RE.sub("", buf)
        emit, buf = _safe_split(buf)
        return emit

    def flush() -> str:
        nonlocal buf
        out = _VISUAL_TAG_RE.sub("", buf)
        buf = ""
        return out

    return feed, flush


# --- Post-processing ---------------------------------------------------------
def _short_year(report_year: str) -> str:
    """'Annual Report 2022-23' -> 'AR 2022-23'."""
    return (report_year or "").replace("Annual Report", "AR").strip() or "?"


def _citations_from_context(context: list[Document]) -> list[dict]:
    """Citations from retrieved context, deduped by (report_year, page)."""
    citations: list[dict] = []
    seen_c: set = set()
    for d in context:
        m = d.metadata
        key = (m.get("report_year"), m.get("page"))
        if key in seen_c or not m.get("report_year"):
            continue
        seen_c.add(key)
        citations.append({
            "report_year": m.get("report_year"),
            "page": m.get("page"),
            "source_pdf": m.get("source_pdf"),
            "label": f"{_short_year(m.get('report_year'))} · p.{m.get('page')}",
        })
    return citations


def _visual_from_doc(doc: Document) -> Optional[dict]:
    m = doc.metadata
    asset = m.get("asset_path")
    if not asset:
        return None
    return {
        "id": m.get("unit_id") or m.get("chunk_id"),
        "type": m.get("content_type"),
        "url": f"/api/assets/{asset}",
        "caption": m.get("caption") or "",
        "page": m.get("page"),
        "source_pdf": m.get("source_pdf"),
    }


def assemble_response(raw: dict[str, Any], query: str) -> dict[str, Any]:
    """Turn raw chain output into {answer_markdown, citations[], visuals[]}."""
    text: str = raw.get("answer", "") or ""
    context: list[Document] = raw.get("context", []) or []

    # Strip [VISUAL: id] tags from the prose (we render visuals structurally).
    tagged_ids = [m.strip() for m in _VISUAL_TAG_RE.findall(text)]
    clean = _VISUAL_TAG_RE.sub("", text)
    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()

    # Honest refusal -> no citations/visuals attached.
    if clean.lower().startswith(REFUSAL.lower().rstrip(".")):
        return {"answer_markdown": REFUSAL, "citations": [], "visuals": []}

    # Resolve LLM-tagged visuals against the retrieved context.
    by_id: dict[str, Document] = {}
    for d in context:
        for key in (d.metadata.get("chunk_id"), d.metadata.get("unit_id")):
            if key:
                by_id[key] = d

    visuals: list[dict] = []
    seen: set = set()
    for vid in tagged_ids:
        doc = by_id.get(vid)
        v = _visual_from_doc(doc) if doc else None
        if v and v["id"] not in seen:
            seen.add(v["id"])
            visuals.append(v)

    # Safety net: if the LLM tagged none, attach the top visual above threshold.
    if not visuals:
        for doc, score in retrieve_visuals(query):
            if score >= settings.VISUAL_SIM_THRESHOLD:
                v = _visual_from_doc(doc)
                if v:
                    visuals.append(v)
                break

    citations = _citations_from_context(context)

    return {"answer_markdown": clean, "citations": citations, "visuals": visuals}


def ask(query: str, chat_history: Optional[list] = None) -> dict[str, Any]:
    """Public entry point: grounded answer + citations + visuals (API contract)."""
    return assemble_response(answer(query, chat_history), query)


async def astream_ask(
    query: str, chat_history: Optional[list] = None
) -> AsyncIterator[dict[str, Any]]:
    """Streaming entry point. Async-yields structured events::

        {"type": "token",    "text": ...}                 # repeated, live answer
        {"type": "metadata", "citations": [], "visuals": []}   # once, at completion
        {"type": "final",    "answer_markdown", "citations", "visuals"}  # internal

    Tokens come straight from ``ChatNVIDIA`` via the retrieval chain's ``.astream``
    (no buffering). ``[VISUAL: id]`` tags are stripped incrementally so they never
    flash. Citations/visuals are resolved from the full answer + context at the end
    by reusing ``assemble_response`` (keeps refusal/visual behavior identical). The
    final event carries the authoritative text for persistence (not sent to client).
    """
    chain = build_chain()
    feed, flush = _make_visual_stripper()
    context: list[Document] = []
    raw_text = ""
    n_tokens = 0
    t0 = time.perf_counter()
    t_retrieval: Optional[float] = None
    t_first_token: Optional[float] = None

    async for chunk in chain.astream({
        "input": query,
        "chat_history": _to_messages(chat_history),
    }):
        ctx = chunk.get("context")
        if ctx and not context:
            context = ctx
            t_retrieval = time.perf_counter() - t0
        piece = chunk.get("answer")
        if piece:
            if t_first_token is None:
                t_first_token = time.perf_counter() - t0
            raw_text += piece
            n_tokens += 1
            emit = feed(piece)
            if emit:
                yield {"type": "token", "text": emit}

    tail = flush()
    if tail:
        yield {"type": "token", "text": tail}

    final = assemble_response({"answer": raw_text, "context": context}, query)
    yield {"type": "metadata",
           "citations": final["citations"], "visuals": final["visuals"]}

    t_end = time.perf_counter() - t0
    gen = (t_end - t_first_token) if t_first_token is not None else None
    log.info(
        "stream timings: retrieval+rerank=%s ttft=%s generation=%s total=%.2fs chunks=%d",
        f"{t_retrieval:.2f}s" if t_retrieval is not None else "n/a",
        f"{t_first_token:.2f}s" if t_first_token is not None else "n/a",
        f"{gen:.2f}s" if gen is not None else "n/a",
        t_end, n_tokens,
    )

    yield {"type": "final",
           "answer_markdown": final["answer_markdown"],
           "citations": final["citations"],
           "visuals": final["visuals"]}


if __name__ == "__main__":  # Manual smoke: python -m backend.chain
    import json
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    res = ask("What is the Rural Infrastructure Development Fund?")
    print(json.dumps(res, indent=2, ensure_ascii=False))
