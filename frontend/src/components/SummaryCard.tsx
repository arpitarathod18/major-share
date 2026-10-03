import { downloadUrl } from '../services/api'

interface Props {
  text: string
  method: string
  downloadPath: string
}

export function SummaryCard({ text, method, downloadPath }: Props) {
  return (
    <section className="animate-rise rounded-2xl border border-[rgba(15,61,62,0.12)] bg-white/75 p-6 shadow-[0_10px_40px_rgba(15,61,62,0.06)]">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl text-[var(--teal-deep)]">Podcast Summary</h2>
          <p className="mt-1 text-xs uppercase tracking-wide text-[rgba(26,36,35,0.55)]">
            Method: {method}
          </p>
        </div>
        <a
          className="rounded-xl bg-[var(--teal)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[var(--teal-deep)]"
          href={downloadUrl(downloadPath)}
          download
        >
          Download summary
        </a>
      </div>
      <p className="mt-5 whitespace-pre-wrap text-[1.05rem] leading-relaxed text-[var(--ink)]">
        {text}
      </p>
    </section>
  )
}
