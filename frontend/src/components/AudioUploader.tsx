import { useCallback, useRef, useState } from 'react'

const ACCEPT = '.mp3,.wav,.m4a,audio/mpeg,audio/wav,audio/mp4,audio/x-m4a'

interface Props {
  file: File | null
  durationLabel: string | null
  disabled?: boolean
  onFileSelected: (file: File) => void
  onRemove: () => void
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export function AudioUploader({
  file,
  durationLabel,
  disabled,
  onFileSelected,
  onRemove,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFiles = useCallback(
    (list: FileList | null) => {
      if (!list || !list.length) return
      onFileSelected(list[0])
    },
    [onFileSelected],
  )

  return (
    <div>
      <span className="mb-2 block text-sm font-semibold tracking-wide text-[var(--teal)]">
        Podcast / audio upload
      </span>
      <div
        className={`rounded-2xl border-2 border-dashed px-5 py-8 text-center transition ${
          dragOver
            ? 'border-[var(--gold)] bg-[rgba(242,193,78,0.12)]'
            : 'border-[rgba(15,61,62,0.22)] bg-white/55'
        } ${disabled ? 'opacity-60' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          if (!disabled) handleFiles(e.dataTransfer.files)
        }}
      >
        {!file ? (
          <>
            <p className="text-lg font-medium text-[var(--teal-deep)]">
              Drag and drop your audio here
            </p>
            <p className="mt-1 text-sm text-[rgba(26,36,35,0.7)]">
              MP3, WAV, or M4A · recommended max 60 minutes
            </p>
            <button
              type="button"
              disabled={disabled}
              className="mt-5 rounded-xl bg-[var(--teal)] px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[var(--teal-deep)] disabled:cursor-not-allowed"
              onClick={() => inputRef.current?.click()}
            >
              Choose file
            </button>
          </>
        ) : (
          <div className="mx-auto max-w-md text-left">
            <p className="truncate text-base font-semibold text-[var(--teal-deep)]">
              {file.name}
            </p>
            <p className="mt-1 text-sm text-[rgba(26,36,35,0.72)]">
              Size: {formatBytes(file.size)}
              {durationLabel ? ` · Duration: ${durationLabel}` : ''}
            </p>
            <button
              type="button"
              disabled={disabled}
              className="mt-4 rounded-xl border border-[rgba(163,59,43,0.35)] px-4 py-2 text-sm font-semibold text-[var(--err)] transition hover:bg-[rgba(163,59,43,0.08)] disabled:cursor-not-allowed"
              onClick={onRemove}
            >
              Remove file
            </button>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          disabled={disabled}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>
    </div>
  )
}
