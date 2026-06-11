// One chat message: role-styled bubble + markdown + visuals + citations.
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import VisualBlock, { CitationChips } from './VisualBlock'

// Bouncing dots shown while the assistant is generating but no text has arrived.
function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" />
    </div>
  )
}

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-slate-900 px-4 py-2 text-sm text-white">
          {message.content}
        </div>
      </div>
    )
  }

  const streaming = message.streaming
  const waiting = streaming && !message.content

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 shadow-sm">
        {waiting ? (
          <TypingDots />
        ) : (
          <div className="prose prose-sm prose-slate max-w-none prose-table:my-2 prose-pre:bg-slate-100">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
              {message.content || ''}
            </ReactMarkdown>
            {streaming && (
              <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-slate-400 align-text-bottom" />
            )}
          </div>
        )}
        <VisualBlock visuals={message.visuals} />
        <CitationChips citations={message.citations} />
      </div>
    </div>
  )
}
