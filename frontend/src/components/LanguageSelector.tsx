import type { LanguageCode } from '../types'

const OPTIONS: { value: LanguageCode; label: string }[] = [
  { value: 'en', label: 'English' },
  { value: 'hi', label: 'Hindi' },
  { value: 'kn', label: 'Kannada' },
  { value: 'te', label: 'Telugu' },
]

interface Props {
  value: LanguageCode
  onChange: (value: LanguageCode) => void
  disabled?: boolean
}

export function LanguageSelector({ value, onChange, disabled }: Props) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm font-semibold tracking-wide text-[var(--teal)]">
        Preferred output language
      </span>
      <select
        className="w-full rounded-xl border border-[rgba(15,61,62,0.18)] bg-white/80 px-4 py-3 text-base outline-none transition focus:border-[var(--gold)] focus:ring-2 focus:ring-[rgba(212,160,23,0.25)] disabled:opacity-60"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value as LanguageCode)}
      >
        {OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  )
}
