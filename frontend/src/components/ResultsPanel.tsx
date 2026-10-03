import type { ProcessResult } from '../types'
import { downloadUrl } from '../services/api'
import { TranscriptViewer } from './TranscriptViewer'
import { SummaryCard } from './SummaryCard'
import { AudioPlayer } from './AudioPlayer'

interface Props {
  result: ProcessResult
  onReset: () => void
}

export function ResultsPanel({ result, onReset }: Props) {
  const languageConfidence =
    result.detected_language_detail?.confidence ?? result.detected_language_probability
  const isCodeSwitched = result.detected_language_detail?.code_switched

  return (
    <div className="space-y-6">
      <section className="animate-rise rounded-2xl border border-[rgba(15,61,62,0.12)] bg-white/80 p-6 shadow-[0_10px_40px_rgba(15,61,62,0.06)]">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-2xl font-bold text-[var(--teal-deep)]">Podcast Processing Analysis</h2>
            <p className="mt-1 text-xs text-gray-500">
              Job ID: <span className="font-mono">{result.job_id}</span> · Completed at{' '}
              {result.processing_time_seconds.toFixed(1)}s
            </p>
          </div>
          <button
            type="button"
            onClick={onReset}
            className="rounded-xl border border-[rgba(15,61,62,0.2)] bg-white px-4 py-2 text-sm font-semibold text-[var(--teal)] shadow-sm transition hover:bg-[rgba(15,61,62,0.05)]"
          >
            Process another podcast
          </button>
        </div>

        <dl className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Info label="Original Filename" value={result.filename} />
          <Info label="Audio Duration" value={result.duration_formatted} />
          <Info
            label="Detected Spoken Language"
            value={`${result.detected_language_name} (${(languageConfidence * 100).toFixed(1)}%)${
              isCodeSwitched ? ' · Multilingual' : ''
            }`}
            badge={isCodeSwitched ? 'Code-Switched' : undefined}
          />
          <Info label="Requested Output Language" value={result.output_language_name} />
          <Info
            label="Voice Diarization"
            value={
              result.transcript.diarized
                ? result.transcript.speaker_count <= 1
                  ? 'Detected speakers: 1'
                  : `Detected speakers: ${result.transcript.speaker_count}`
                : 'Not identified'
            }
            subvalue={
              result.transcript.diarized
                ? result.transcript.speaker_count <= 1
                  ? 'Single-speaker recording (labels hidden)'
                  : result.transcript.method
                  ? `Method: ${result.transcript.method}`
                  : 'Multi-speaker audio'
                : undefined
            }
          />
          <Info
            label="Quality Validation"
            value={result.validation?.passed ? 'Passed ✓' : 'Completed with warnings'}
            subvalue={`${result.transcript.segments.length} segments aligned`}
          />
        </dl>

        {result.speakers && result.speakers.length >= 2 && (
          <div className="mt-4 flex flex-wrap items-center gap-2 rounded-xl bg-[rgba(15,61,62,0.03)] px-4 py-2.5">
            <span className="text-xs font-semibold uppercase text-gray-500">Active Speakers:</span>
            {result.speakers.map((spk) => (
              <span
                key={spk}
                className="inline-flex items-center gap-1 rounded-md bg-white px-2.5 py-1 text-xs font-bold text-[var(--teal-deep)] shadow-sm border border-gray-200"
              >
                🎙 {spk}
              </span>
            ))}
          </div>
        )}

        {result.warnings?.length > 0 && (
          <div className="mt-5 space-y-2">
            {result.warnings.map((w) => (
              <p
                key={w}
                className="rounded-xl border border-[rgba(212,160,23,0.35)] bg-[rgba(242,193,78,0.12)] px-4 py-3 text-xs text-[var(--teal-deep)]"
              >
                ⚠️ {w}
              </p>
            ))}
          </div>
        )}

        <div className="mt-6 flex flex-wrap gap-3 border-t border-gray-100 pt-4">
          <a
            className="inline-flex items-center gap-2 rounded-xl bg-[var(--teal)] px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:brightness-110"
            href={downloadUrl(result.downloads.transcript)}
            download
          >
            📥 Download Transcript (.txt)
          </a>
          <a
            className="inline-flex items-center gap-2 rounded-xl bg-[var(--coral)] px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:brightness-110"
            href={downloadUrl(result.downloads.summary)}
            download
          >
            📄 Download Summary (.txt)
          </a>
          {result.downloads.audio && (
            <a
              className="inline-flex items-center gap-2 rounded-xl border border-[var(--teal)] bg-white px-4 py-2.5 text-sm font-semibold text-[var(--teal)] shadow-sm transition hover:bg-[rgba(15,61,62,0.04)]"
              href={downloadUrl(result.downloads.audio)}
              download
            >
              🎧 Download Audio Summary (.mp3)
            </a>
          )}
        </div>
      </section>

      <TranscriptViewer
        segments={result.transcript.segments}
        diarized={result.transcript.diarized}
        message={result.transcript.message}
        originalAudioUrl={result.original_audio_url}
      />

      <SummaryCard
        text={result.summary.text}
        method={result.summary.method}
        downloadPath={result.downloads.summary}
      />

      <AudioPlayer
        available={result.audio_summary.available}
        url={result.audio_summary.url}
        message={result.audio_summary.message}
      />
    </div>
  )
}

function Info({
  label,
  value,
  subvalue,
  badge,
}: {
  label: string
  value: string
  subvalue?: string
  badge?: string
}) {
  return (
    <div className="rounded-xl bg-[rgba(15,61,62,0.04)] p-3.5 border border-[rgba(15,61,62,0.06)]">
      <div className="flex items-center justify-between">
        <dt className="text-[11px] font-bold uppercase tracking-wider text-[rgba(26,36,35,0.55)]">
          {label}
        </dt>
        {badge && (
          <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800">
            {badge}
          </span>
        )}
      </div>
      <dd className="mt-1 text-sm font-semibold text-[var(--teal-deep)]">{value}</dd>
      {subvalue && <p className="mt-0.5 text-[11px] text-gray-500">{subvalue}</p>}
    </div>
  )
}
