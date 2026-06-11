// Single-page chatbot: sidebar + chat panel; send flow + session reopen.
import { useCallback, useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'
import {
  listSessions, getSession, sendChatStream, deleteSession,
  getStoredSessionId, setStoredSessionId, clearStoredSessionId,
} from './utils/api'

export default function App() {
  const [sessions, setSessions] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await listSessions())
    } catch {
      /* sidebar refresh is non-critical */
    }
  }, [])

  // On mount: load the session list and restore the last open session.
  useEffect(() => {
    refreshSessions()
    const stored = getStoredSessionId()
    if (!stored) return
    getSession(stored)
      .then((s) => {
        setActiveId(s.id)
        setMessages(s.messages || [])
      })
      .catch(() => clearStoredSessionId()) // stale/deleted id -> start fresh
  }, [refreshSessions])

  const handleNewChat = () => {
    setActiveId(null)
    setMessages([])
    setError(null)
    clearStoredSessionId()
  }

  const handleSelect = async (id) => {
    if (id === activeId) return
    setError(null)
    try {
      const s = await getSession(id)
      setActiveId(s.id)
      setMessages(s.messages || [])
      setStoredSessionId(s.id)
    } catch {
      setError('Could not open that conversation.')
      refreshSessions()
    }
  }

  const handleDelete = async (id) => {
    try {
      await deleteSession(id)
    } catch {
      /* ignore */
    }
    if (id === activeId) handleNewChat()
    refreshSessions()
  }

  // Immutably update the last (streaming) assistant message in place.
  const updateLastAssistant = (patch) =>
    setMessages((prev) => {
      if (!prev.length) return prev
      const next = prev.slice()
      const last = next[next.length - 1]
      next[next.length - 1] =
        typeof patch === 'function' ? patch(last) : { ...last, ...patch }
      return next
    })

  const handleSend = async (message) => {
    setError(null)
    // User message + an empty assistant placeholder that fills as tokens arrive.
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: message },
      { role: 'assistant', content: '', visuals: [], citations: [], streaming: true },
    ])
    setLoading(true)

    await sendChatStream(message, activeId, {
      onMeta: ({ session_id }) => {
        if (!activeId && session_id) {
          setActiveId(session_id)
          setStoredSessionId(session_id)
        }
      },
      onToken: (text) =>
        updateLastAssistant((m) => ({ ...m, content: m.content + text })),
      onMetadata: ({ citations, visuals }) =>
        updateLastAssistant({ citations: citations || [], visuals: visuals || [] }),
      onDone: () => {
        updateLastAssistant((m) => ({ ...m, streaming: false }))
        setLoading(false)
        refreshSessions() // pick up the new/auto-titled session + reordering
      },
      onError: (msg) => {
        setError(msg || 'Something went wrong.')
        // Drop the placeholder if nothing streamed; otherwise keep what we have.
        setMessages((prev) => {
          const last = prev[prev.length - 1]
          if (last?.role === 'assistant' && last.streaming && !last.content) {
            return prev.slice(0, -1)
          }
          return prev.map((m, i) =>
            i === prev.length - 1 ? { ...m, streaming: false } : m,
          )
        })
        setLoading(false)
      },
    })
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        onSelect={handleSelect}
        onNew={handleNewChat}
        onDelete={handleDelete}
      />
      <ChatPanel
        messages={messages}
        loading={loading}
        error={error}
        onSend={handleSend}
        onDismissError={() => setError(null)}
      />
    </div>
  )
}
