// Single-page chatbot: sidebar + chat panel; send flow + session reopen
// (Tasks 5.5 + 5.7).
import { useCallback, useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'
import {
  listSessions, getSession, sendChat, deleteSession,
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

  const handleSend = async (message) => {
    setError(null)
    setMessages((prev) => [...prev, { role: 'user', content: message }])
    setLoading(true)
    try {
      const data = await sendChat(message, activeId)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer_markdown,
          visuals: data.visuals || [],
          citations: data.citations || [],
        },
      ])
      if (!activeId) {
        setActiveId(data.session_id)
        setStoredSessionId(data.session_id)
      }
      refreshSessions() // pick up the new/auto-titled session + reordering
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
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
