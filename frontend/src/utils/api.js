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

// visuals[].url is already "/api/assets/..."; the proxy resolves it directly.
export function assetUrl(url) {
  return url
}
