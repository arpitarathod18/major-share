import { useEffect, useMemo, useRef, useState } from 'react'
import type { TranscriptSegment } from '../types'
import { downloadUrl } from '../services/api'

interface Props {
  segments: TranscriptSegment[]
  diarized: boolean
  message: string | null
  originalAudioUrl?: string
}

const SPEAKER_THEMES: Record<
  number,
  { border: string; bg: string; badge: string; text: string }
> = {
  0: {
    border: 'border-l-[var(--teal)]',
    bg: 'bg-[rgba(15,61,62,0.04)]',
    badge: 'bg-[var(--teal)] text-white',
    text: 'text-[var(--teal-deep)]',
  },
  1: {
    border: 'border-l-[var(--coral)]',
    bg: 'bg-[rgba(196,92,38,0.05)]',
    badge: 'bg-[var(--coral)] text-white',
    text: 'text-[#8a3311]',
  },
  2: {
    border: 'border-l-[var(--gold)]',
    bg: 'bg-[rgba(212,160,23,0.06)]',
    badge: 'bg-[var(--gold)] text-[var(--teal-deep)]',
    text: 'text-[#6f4f03]',
  },
  3: {
    border: 'border-l-[#2f6b8a]',
    bg: 'bg-[rgba(47,107,138,0.05)]',
    badge: 'bg-[#2f6b8a] text-white',
    text: 'text-[#1a445a]',
  },
}

const LANGUAGE_BADGES: Record<string, string> = {
  en: 'bg-blue-100 text-blue-800 border-blue-200',
  hi: 'bg-orange-100 text-orange-800 border-orange-200',
  kn: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  te: 'bg-purple-100 text-purple-800 border-purple-200',
}

function fmt(seconds: number): string {
  const total = Math.max(0, Math.round(seconds * 1000))
  const h = Math.floor(total / 3_600_000)
  const m = Math.floor((total % 3_600_000) / 60_000)
  const s = Math.floor((total % 60_000) / 1000)
  const ms = total % 1000
  if (h > 0) {
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`
  }
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`
}

function getSpeakerTheme(label: string | null) {
  if (!label) {
    return {
      border: 'border-l-gray-400',
      bg: 'bg-gray-50',
      badge: 'bg-gray-500 text-white',
      text: 'text-gray-700',
    }
  }
  const match = label.match(/(\d+)/)
  const idx = match ? parseInt(match[1], 10) % 4 : 0
  return SPEAKER_THEMES[idx] || SPEAKER_THEMES[0]
}

export function TranscriptViewer({ segments, diarized, message, originalAudioUrl }: Props) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const segmentRefs = useRef<(HTMLElement | null)[]>([])
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [selectedSpeaker, setSelectedSpeaker] = useState<string>('all')

  const activeIndex = useMemo(
    () => segments.findIndex((seg) => currentTime >= seg.start && currentTime < seg.end),
    [currentTime, segments],
  )

  const distinctSpeakers = useMemo(() => {
    const set = new Set<string>()
    for (const s of segments) {
      if (s.speaker) set.add(s.speaker)
    }
    return Array.from(set).sort()
  }, [segments])

  const isMultiSpeaker = distinctSpeakers.length >= 2

  const filteredSegments = useMemo(() => {
    if (!isMultiSpeaker || selectedSpeaker === 'all') return segments
    return segments.filter((s) => s.speaker === selectedSpeaker)
  }, [segments, selectedSpeaker, isMultiSpeaker])

  const seek = (start: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(0, start)
      void audioRef.current.play()
      setIsPlaying(true)
    }
  }

  // Smooth scroll active segment into view when playing
  useEffect(() => {
    if (isPlaying && activeIndex >= 0 && segmentRefs.current[activeIndex]) {
      segmentRefs.current[activeIndex]?.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      })
    }
  }, [activeIndex, isPlaying])

  return (
    <section className="animate-rise rounded-2xl border border-[rgba(15,61,62,0.12)] bg-white/85 p-6 shadow-[0_10px_40px_rgba(15,61,62,0.06)]">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-[var(--teal-deep)]">
            {isMultiSpeaker ? 'Speaker-Timestamped Transcript' : 'Timestamped Transcript'}
          </h2>
          <p className="mt-1 text-sm text-[rgba(26,36,35,0.68)]">
            Click any segment or timestamp to seek audio directly to that point.
          </p>
        </div>

        {isMultiSpeaker && (
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase text-gray-500">Filter Speaker:</span>
            <select
              value={selectedSpeaker}
              onChange={(e) => setSelectedSpeaker(e.target.value)}
              className="rounded-lg border border-gray-300 bg-white px-2.5 py-1.5 text-xs font-medium text-gray-700 shadow-sm focus:border-[var(--teal)] focus:outline-none"
            >
              <option value="all">All Speakers ({distinctSpeakers.length})</option>
              {distinctSpeakers.map((spk) => (
                <option key={spk} value={spk}>
                  {spk}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {originalAudioUrl && (
        <div className="mt-4 rounded-xl border border-[rgba(15,61,62,0.1)] bg-[rgba(15,61,62,0.02)] p-3">
          <div className="mb-2 flex items-center justify-between text-xs font-semibold text-[var(--teal)]">
            <span>🎧 Original Audio (Time Synchronized)</span>
            <span>Current: {fmt(currentTime)}</span>
          </div>
          <audio
            ref={audioRef}
            className="w-full"
            controls
            preload="metadata"
            src={downloadUrl(originalAudioUrl)}
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
          />
        </div>
      )}

      {!diarized && message && (
        <div className="mt-4 rounded-xl border border-[rgba(212,160,23,0.35)] bg-[rgba(242,193,78,0.12)] px-4 py-3 text-sm text-[var(--teal-deep)]">
          <p className="font-semibold">Speaker Diarization Notice:</p>
          <p className="mt-0.5 text-xs">{message}</p>
        </div>
      )}

      <div className="mt-5 max-h-[36rem] space-y-3.5 overflow-y-auto pr-1">
        {filteredSegments.map((seg, i) => {
          const theme = isMultiSpeaker ? getSpeakerTheme(seg.speaker) : {
            border: 'border-l-[var(--teal)]',
            bg: 'bg-[rgba(15,61,62,0.03)]',
            badge: '',
            text: 'text-[var(--teal-deep)]',
          }
          const isActive = activeIndex === i
          const langClass =
            LANGUAGE_BADGES[seg.language] || 'bg-gray-100 text-gray-800 border-gray-200'

          return (
            <article
              key={`${seg.start}-${i}`}
              ref={(el) => {
                segmentRefs.current[i] = el
              }}
              onClick={() => seek(seg.start)}
              className={`group cursor-pointer rounded-r-xl border-l-4 px-4 py-3.5 transition-all duration-200 hover:shadow-md ${
                theme.border
              } ${theme.bg} ${
                isActive
                  ? 'ring-2 ring-[var(--gold)] shadow-md bg-[rgba(242,193,78,0.15)] scale-[1.002]'
                  : 'hover:brightness-95'
              }`}
            >
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs font-semibold tracking-wide">
                <div className="flex flex-wrap items-center gap-2">
                  {isMultiSpeaker && (
                    <span
                      className={`inline-flex items-center rounded-md px-2 py-0.5 font-mono text-[11px] font-bold uppercase shadow-sm ${theme.badge}`}
                    >
                      🎙 {seg.speaker || 'UNASSIGNED'}
                    </span>
                  )}
                  <span className="font-mono text-gray-600">
                    {fmt(seg.start)} → {fmt(seg.end)}
                  </span>
                  <span className="rounded bg-gray-200/80 px-1.5 py-0.5 text-[10px] text-gray-700">
                    {seg.duration.toFixed(3)}s
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <span
                    className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold ${langClass}`}
                  >
                    {seg.language_name}
                  </span>
                  <span className="opacity-0 transition-opacity group-hover:opacity-100 text-[11px] font-medium text-[var(--teal)]">
                    ▶ Play
                  </span>
                </div>
              </div>

              <p className="text-[1.05rem] leading-relaxed text-[var(--ink)] antialiased font-sans">
                {seg.text}
              </p>
            </article>
          )
        })}

        {!filteredSegments.length && (
          <p className="py-8 text-center text-sm text-[rgba(26,36,35,0.65)]">
            No transcript segments available for the selected filter.
          </p>
        )}
      </div>
    </section>
  )
}
