// Scrollable conversation + input + loading/error UX (Task 5.4/5.5).
// The streaming assistant placeholder (in App.handleSend) renders its own
// in-bubble typing indicator, so no separate global indicator is needed.
import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'
import ChatInput from './ChatInput'

export default function ChatPanel({ messages, loading, error, onSend, onDismissError }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  return (
    <main className="flex h-full flex-1 flex-col bg-slate-50">
      <header className="border-b border-slate-200 bg-white px-6 py-3">
        <h1 className="text-base font-semibold text-slate-800">Annual Reports Assistant</h1>
        <p className="text-xs text-slate-400">Answers grounded in the annual report PDFs, with citations.</p>
      </header>

      <div className="scroll-thin flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.length === 0 && !loading && (
            <div className="mt-20 text-center text-slate-400">
              <p className="text-lg font-medium text-slate-500">Ask a question to get started</p>
              <p className="mt-1 text-sm">
                e.g. “What is the Rural Infrastructure Development Fund?” or “Total borrowings from the twenty largest lenders?”
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <MessageBubble key={m.id ?? i} message={m} />
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      {error && (
        <div className="mx-auto mb-2 flex max-w-3xl items-center justify-between gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
          <span>{error}</span>
          <button onClick={onDismissError} className="font-medium text-red-500 hover:text-red-700">✕</button>
        </div>
      )}

      <ChatInput onSend={onSend} disabled={loading} />
    </main>
  )
}
