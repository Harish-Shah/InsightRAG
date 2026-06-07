// Session list + New Chat (Task 5.3).
export default function Sidebar({ sessions, activeId, onSelect, onNew, onDelete }) {
  return (
    <aside className="flex h-full w-72 flex-col border-r border-slate-200 bg-white">
      <div className="p-3">
        <button
          onClick={onNew}
          className="w-full rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700"
        >
          + New Chat
        </button>
      </div>

      <div className="scroll-thin flex-1 overflow-y-auto px-2 pb-3">
        {sessions.length === 0 ? (
          <p className="px-2 py-6 text-center text-sm text-slate-400">
            No conversations yet. Start a new chat.
          </p>
        ) : (
          <ul className="space-y-1">
            {sessions.map((s) => (
              <li key={s.id}>
                <div
                  onClick={() => onSelect(s.id)}
                  className={`group flex cursor-pointer items-center justify-between rounded-lg px-3 py-2 text-sm transition ${
                    s.id === activeId
                      ? 'bg-slate-200 text-slate-900'
                      : 'text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  <span className="truncate" title={s.title}>{s.title}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete(s.id)
                    }}
                    className="ml-2 hidden shrink-0 rounded p-1 text-slate-400 hover:text-red-600 group-hover:block"
                    title="Delete conversation"
                    aria-label="Delete conversation"
                  >
                    ✕
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="border-t border-slate-200 p-3 text-xs text-slate-400">
        Annual Reports RAG · grounded answers with citations
      </div>
    </aside>
  )
}
