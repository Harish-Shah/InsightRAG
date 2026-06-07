// One chat message: role-styled bubble + markdown + visuals + citations (Task 5.4/5.6).
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import VisualBlock, { CitationChips } from './VisualBlock'

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

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 shadow-sm">
        <div className="prose prose-sm prose-slate max-w-none prose-table:my-2 prose-pre:bg-slate-100">
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
            {message.content || ''}
          </ReactMarkdown>
        </div>
        <VisualBlock visuals={message.visuals} />
        <CitationChips citations={message.citations} />
      </div>
    </div>
  )
}
