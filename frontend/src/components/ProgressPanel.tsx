const STAGES = [
  { key: 'upload', label: 'Uploading audio' },
  { key: 'language_detection', label: 'Detecting language' },
  { key: 'transcription', label: 'Transcribing audio' },
  { key: 'diarization', label: 'Identifying speakers' },
  { key: 'summarization', label: 'Generating summary' },
  { key: 'tts', label: 'Generating audio summary' },
  { key: 'completed', label: 'Completed' },
]

interface Props {
  stage: string
  stageLabel: string
  progress: number
  elapsedSeconds: number | null
}

function formatElapsed(seconds: number | null): string {
  if (seconds == null) return '0s'
  const s = Math.max(0, Math.floor(seconds))
  const m = Math.floor(s / 60)
  const r = s % 60
  return m > 0 ? `${m}m ${r}s` : `${r}s`
}

export function ProgressPanel({ stage, stageLabel, progress, elapsedSeconds }: Props) {
  const activeIndex = Math.max(
    0,
    STAGES.findIndex((s) => s.key === stage),
  )

  return (
    <section className="animate-rise rounded-2xl border border-[rgba(15,61,62,0.12)] bg-white/75 p-6 shadow-[0_10px_40px_rgba(15,61,62,0.06)]">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="spinner" aria-hidden />
          <div>
            <h2 className="text-xl font-semibold text-[var(--teal-deep)]">Processing podcast</h2>
            <p className="text-sm text-[rgba(26,36,35,0.7)]">{stageLabel || 'Working…'}</p>
          </div>
        </div>
        <p className="text-sm font-medium text-[var(--teal)]">
          Elapsed: {formatElapsed(elapsedSeconds)}
        </p>
      </div>

      <div className="mt-5 h-2.5 overflow-hidden rounded-full bg-[rgba(15,61,62,0.1)]">
        <div
          className="h-full rounded-full bg-[linear-gradient(90deg,var(--teal),var(--gold))] transition-all duration-500"
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
        />
      </div>
      <p className="mt-2 text-right text-sm font-semibold text-[var(--teal)]">{progress}%</p>

      <ol className="mt-5 space-y-2">
        {STAGES.map((s, idx) => {
          const done = idx < activeIndex || stage === 'completed'
          const current = idx === activeIndex && stage !== 'completed'
          return (
            <li
              key={s.key}
              className={`flex items-center gap-3 rounded-xl px-3 py-2 text-sm ${
                current
                  ? 'bg-[rgba(242,193,78,0.16)] font-semibold text-[var(--teal-deep)]'
                  : done
                    ? 'text-[var(--ok)]'
                    : 'text-[rgba(26,36,35,0.45)]'
              }`}
            >
              <span
                className={`inline-flex h-6 w-6 items-center justify-center rounded-full text-xs ${
                  current
                    ? 'animate-pulse-soft bg-[var(--gold-soft)] text-[var(--teal-deep)]'
                    : done
                      ? 'bg-[rgba(31,122,77,0.15)]'
                      : 'bg-[rgba(15,61,62,0.08)]'
                }`}
              >
                {done ? '✓' : idx + 1}
              </span>
              {s.label}
            </li>
          )
        })}
      </ol>
    </section>
  )
}
