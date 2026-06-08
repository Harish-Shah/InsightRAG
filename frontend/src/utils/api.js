// Centralized backend calls + current-session persistence (Task 5.2).
// Same-origin "/api" is proxied to the FastAPI backend by Vite in dev.
import axios from 'axios'

const client = axios.create({ baseURL: '/api', timeout: 120000 })

const SESSION_KEY = 'rag.session_id'

// --- current session id (localStorage) ---
export function getStoredSessionId() {
  return localStorage.getItem(SESSION_KEY) || null
}
export function setStoredSessionId(id) {
  if (id) localStorage.setItem(SESSION_KEY, id)
}
export function clearStoredSessionId() {
  localStorage.removeItem(SESSION_KEY)
}

// Turn an axios error into a short, user-facing message.
function friendly(err, fallback) {
  const detail = err?.response?.data?.detail
  if (detail) return typeof detail === 'string' ? detail : fallback
  if (err?.code === 'ECONNABORTED') return 'The request timed out. Please try again.'
  if (err?.message === 'Network Error') return 'Cannot reach the server. Is the backend running?'
  return fallback
}

// --- API helpers ---
export async function listSessions() {
  const { data } = await client.get('/sessions')
  return data
}

export async function createSession(title) {
  const { data } = await client.post('/sessions', { title: title ?? null })
  return data
}

export async function getSession(id) {
  const { data } = await client.get(`/sessions/${id}`)
  return data
}

export async function deleteSession(id) {
  await client.delete(`/sessions/${id}`)
}

export async function sendChat(message, sessionId) {
  try {
    const { data } = await client.post('/chat', {
      session_id: sessionId ?? null,
      message,
    })
    return data
  } catch (err) {
    throw new Error(friendly(err, 'The assistant could not answer. Please try again.'))
  }
}

// Streaming chat over SSE (fetch + ReadableStream; EventSource can't POST a body).
// Dispatches parsed events to handlers as they arrive:
//   onMeta({session_id}), onToken(text), onMetadata({citations, visuals}),
//   onDone(), onError(message).
export async function sendChatStream(
  message,
  sessionId,
  { onMeta, onToken, onMetadata, onDone, onError } = {},
) {
  let res
  try {
    res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId ?? null, message }),
    })
  } catch {
    onError?.('Cannot reach the server. Is the backend running?')
    return
  }
  if (!res.ok || !res.body) {
    let detail = 'The assistant could not answer. Please try again.'
    try {
      const body = await res.json()
      if (typeof body?.detail === 'string') detail = body.detail
    } catch {
      /* non-JSON error body */
    }
    onError?.(detail)
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''

  const dispatch = (evt) => {
    switch (evt.type) {
      case 'meta': onMeta?.(evt); break
      case 'token': onToken?.(evt.text); break
      case 'metadata': onMetadata?.(evt); break
      case 'done': onDone?.(); break
      case 'error': onError?.(evt.detail || 'Something went wrong.'); break
      default: /* ignore unknown event types */
    }
  }

  // SSE frames are separated by a blank line; each carries one `data: <json>`.
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, sep)
      buf = buf.slice(sep + 2)
      const line = frame.split('\n').find((l) => l.startsWith('data:'))
      if (!line) continue
      try {
        dispatch(JSON.parse(line.slice(5).trim()))
      } catch {
        /* skip malformed frame */
      }
    }
  }
}

// visuals[].url is already "/api/assets/..."; the proxy resolves it directly.
export function assetUrl(url) {
  return url
}
