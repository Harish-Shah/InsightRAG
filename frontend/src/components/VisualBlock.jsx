// Returned visuals (images + click-to-zoom) and citation chips.
import { useEffect, useState } from 'react'
import { assetUrl } from '../utils/api'

function ImageCard({ visual, onZoom }) {
  const [broken, setBroken] = useState(false)
  if (broken) {
    return (
      <div className="flex h-32 w-full items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 text-xs text-slate-400">
        image unavailable
      </div>
    )
  }
  return (
    <figure className="overflow-hidden rounded-lg border border-slate-200 bg-white">
      <img
        src={assetUrl(visual.url)}
        alt={visual.caption || visual.type}
        loading="lazy"
        onClick={() => onZoom(visual)}
        onError={() => setBroken(true)}
        className="max-h-56 w-full cursor-zoom-in object-contain bg-white"
      />
      <figcaption className="truncate px-2 py-1 text-[11px] text-slate-500" title={visual.caption}>
        {visual.type}{visual.caption ? ` · ${visual.caption}` : ''}
      </figcaption>
    </figure>
  )
}

function ZoomModal({ visual, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
    >
      <div className="max-h-[90vh] max-w-[90vw] overflow-auto" onClick={(e) => e.stopPropagation()}>
        <img
          src={assetUrl(visual.url)}
          alt={visual.caption || visual.type}
          className="max-h-[85vh] max-w-[90vw] rounded bg-white object-contain"
        />
        {visual.caption && (
          <p className="mt-2 text-center text-sm text-slate-200">{visual.caption}</p>
        )}
      </div>
      <button
        onClick={onClose}
        className="absolute right-4 top-4 rounded-full bg-white/90 px-3 py-1 text-sm font-medium text-slate-800"
      >
        Close ✕
      </button>
    </div>
  )
}

export function CitationChips({ citations }) {
  if (!citations?.length) return null
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {citations.map((c, i) => (
        <span
          key={`${c.report_year}-${c.page}-${i}`}
          className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600"
          title={c.source_pdf || ''}
        >
          {c.label || `${c.report_year} · p.${c.page}`}
        </span>
      ))}
    </div>
  )
}

export default function VisualBlock({ visuals }) {
  const [zoomed, setZoomed] = useState(null)
  if (!visuals?.length) return null
  return (
    <>
      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {visuals.map((v) => (
          <ImageCard key={v.id} visual={v} onZoom={setZoomed} />
        ))}
      </div>
      {zoomed && <ZoomModal visual={zoomed} onClose={() => setZoomed(null)} />}
    </>
  )
}
