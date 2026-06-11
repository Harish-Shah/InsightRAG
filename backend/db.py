"""SQLite persistence for sessions + messages.

Durable storage so chats reopen with full content (text, visuals, citations).
Uses stdlib ``sqlite3`` with a fresh connection per call (safe under FastAPI's
threadpool for this single-user demo). Visuals and citations are stored as JSON
text and round-tripped on read.

Schema::

    sessions(id TEXT PK, title, created_at, updated_at)
    messages(id INTEGER PK, session_id FK -> sessions ON DELETE CASCADE,
             role, content, visuals_json, citations_json, created_at)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Optional

from backend.config import settings


def _now() -> str:
    """Current UTC time as an ISO-8601 string (sortable)."""
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    settings.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist (called on startup)."""
    with closing(_connect()) as conn, conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL,
                role          TEXT NOT NULL,
                content       TEXT NOT NULL,
                visuals_json  TEXT,
                citations_json TEXT,
                created_at    TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);
            """
        )


# --- Sessions ----------------------------------------------------------------
def create_session(title: str) -> str:
    sid = uuid.uuid4().hex
    ts = _now()
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (sid, title, ts, ts),
        )
    return sid


def list_sessions() -> list[dict[str, Any]]:
    """All sessions, newest-updated first."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions "
            "ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[dict[str, Any]]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def get_session_with_messages(session_id: str) -> Optional[dict[str, Any]]:
    """Full session incl. messages with parsed visuals/citations, or None."""
    session = get_session(session_id)
    if session is None:
        return None
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, role, content, visuals_json, citations_json, created_at "
            "FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    session["messages"] = [
        {
            "id": r["id"],
            "role": r["role"],
            "content": r["content"],
            "visuals": json.loads(r["visuals_json"]) if r["visuals_json"] else [],
            "citations": json.loads(r["citations_json"]) if r["citations_json"] else [],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    return session


def touch_session(session_id: str) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id)
        )


def delete_session(session_id: str) -> bool:
    """Delete a session (messages cascade). Returns True if a row was removed."""
    with closing(_connect()) as conn, conn:
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0


# --- Messages ----------------------------------------------------------------
def add_message(session_id: str, role: str, content: str,
                visuals: Optional[list] = None,
                citations: Optional[list] = None) -> int:
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO messages "
            "(session_id, role, content, visuals_json, citations_json, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                session_id, role, content,
                json.dumps(visuals or [], ensure_ascii=False),
                json.dumps(citations or [], ensure_ascii=False),
                _now(),
            ),
        )
        return int(cur.lastrowid)
