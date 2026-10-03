import { downloadUrl } from '../services/api'

interface Props {
  available: boolean
  url: string | null
  message: string | null
}

export function AudioPlayer({ available, url, message }: Props) {
  if (!available || !url) {
    return (
      <section className="animate-rise rounded-2xl border border-[rgba(15,61,62,0.12)] bg-white/75 p-6 shadow-[0_10px_40px_rgba(15,61,62,0.06)]">
        <h2 className="text-2xl text-[var(--teal-deep)]">Audio Summary</h2>
        <p className="mt-3 rounded-xl border border-[rgba(163,59,43,0.25)] bg-[rgba(163,59,43,0.06)] px-4 py-3 text-sm text-[var(--err)]">
          {message || 'Audio summary is not available.'}
        </p>
      </section>
    )
  }

  const src = downloadUrl(url)

  return (
    <section className="animate-rise rounded-2xl border border-[rgba(15,61,62,0.12)] bg-white/75 p-6 shadow-[0_10px_40px_rgba(15,61,62,0.06)]">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <h2 className="text-2xl text-[var(--teal-deep)]">Audio Summary</h2>
        <a
          className="rounded-xl border border-[rgba(15,61,62,0.2)] px-4 py-2 text-sm font-semibold text-[var(--teal)] transition hover:bg-[rgba(15,61,62,0.05)]"
          href={src}
          download
        >
          Download audio
        </a>
      </div>
      <audio className="mt-5 w-full" controls src={src} preload="metadata">
        Your browser does not support the audio element.
      </audio>
    </section>
  )
}
