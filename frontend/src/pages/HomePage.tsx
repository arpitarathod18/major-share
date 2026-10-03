import { useCallback, useEffect, useRef, useState } from 'react'
import { AudioUploader } from '../components/AudioUploader'
import { LanguageSelector } from '../components/LanguageSelector'
import { ProgressPanel } from '../components/ProgressPanel'
import { ResultsPanel } from '../components/ResultsPanel'
import {
  fetchHealth,
  getErrorMessage,
  getResult,
  getStatus,
  startProcessing,
  uploadAudio,
} from '../services/api'
import type { HealthResponse, LanguageCode, ProcessResult, StatusResponse } from '../types'

type AppPhase = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed'

const ALLOWED_EXT = ['.mp3', '.wav', '.m4a']

export function HomePage() {
  const [language, setLanguage] = useState<LanguageCode>('en')
  const [file, setFile] = useState<File | null>(null)
  const [localDuration, setLocalDuration] = useState<string | null>(null)
  const [phase, setPhase] = useState<AppPhase>('idle')
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [result, setResult] = useState<ProcessResult | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const pollRef = useRef<number | null>(null)

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() =>
        setHealth({
          backend: 'unreachable',
          whisper: 'unknown',
          ffmpeg: 'unknown',
          gemini: 'unknown',
          diarization: 'unknown',
          tts: 'unknown',
        }),
      )
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current)
    }
  }, [])

  const clearPoll = () => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const resetAll = () => {
    clearPoll()
    setFile(null)
    setLocalDuration(null)
    setPhase('idle')
    setError(null)
    setStatus(null)
    setResult(null)
  }

  const onFileSelected = useCallback((selected: File) => {
    setError(null)
    setResult(null)
    setStatus(null)
    setPhase('idle')

    const lower = selected.name.toLowerCase()
    const ok = ALLOWED_EXT.some((ext) => lower.endsWith(ext))
    if (!ok) {
      setFile(null)
      setLocalDuration(null)
      setError('Unsupported file type. Please upload an MP3, WAV, or M4A audio file.')
      return
    }

    setFile(selected)
    setLocalDuration(null)

    const url = URL.createObjectURL(selected)
    const audio = new Audio()
    audio.preload = 'metadata'
    audio.src = url
    audio.onloadedmetadata = () => {
      const d = audio.duration
      if (Number.isFinite(d)) {
        const h = Math.floor(d / 3600)
        const m = Math.floor((d % 3600) / 60)
        const s = Math.floor(d % 60)
        setLocalDuration(
          `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`,
        )
      }
      URL.revokeObjectURL(url)
    }
    audio.onerror = () => {
      URL.revokeObjectURL(url)
    }
  }, [])

  const pollJob = (jobId: string) => {
    clearPoll()
    pollRef.current = window.setInterval(async () => {
      try {
        const st = await getStatus(jobId)
        setStatus(st)
        if (st.status === 'completed') {
          clearPoll()
          const res = await getResult(jobId)
          setResult(res)
          setPhase('completed')
        } else if (st.status === 'failed') {
          clearPoll()
          setPhase('failed')
          setError(st.error || 'Processing failed.')
        }
      } catch (err) {
        clearPoll()
        setPhase('failed')
        setError(getErrorMessage(err, 'Could not check processing status.'))
      }
    }, 1500)
  }

  const handleProcess = async () => {
    if (!file) {
      setError('Please select an audio file first.')
      return
    }
    setError(null)
    setResult(null)
    setPhase('uploading')
    setStatus({
      success: true,
      job_id: '',
      status: 'processing',
      stage: 'upload',
      stage_label: 'Uploading audio',
      progress: 5,
      elapsed_seconds: 0,
    })

    try {
      const uploaded = await uploadAudio(file)
      if (!uploaded.success) {
        throw new Error(uploaded.error || 'Upload failed.')
      }
      if (uploaded.duration_formatted) {
        setLocalDuration(uploaded.duration_formatted)
      }

      setPhase('processing')
      const started = await startProcessing(uploaded.upload_id, language)
      if (!started.success || !started.job_id) {
        throw new Error(started.error || 'Could not start processing.')
      }
      setStatus((prev) =>
        prev
          ? { ...prev, job_id: started.job_id, stage_label: 'Detecting language', progress: 15 }
          : prev,
      )
      pollJob(started.job_id)
    } catch (err) {
      setPhase('failed')
      setError(getErrorMessage(err, 'Upload or processing could not be started.'))
    }
  }

  const busy = phase === 'uploading' || phase === 'processing'

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:py-12">
      <header className="animate-rise mb-10 text-center">
        <p className="brand text-4xl font-bold leading-tight text-[var(--teal-deep)] sm:text-5xl">
          Meri Awaaz Hi Pehchan Hai
        </p>
        <p className="mt-3 text-base text-[rgba(26,36,35,0.72)] sm:text-lg">
          Multilingual Podcast Processing System for India
        </p>
        {health && (
          <p className="mt-4 text-xs text-[rgba(26,36,35,0.55)]">
            Backend: {health.backend} · Whisper:{" "}
            {typeof health.whisper === "string"
              ? health.whisper
              : health.whisper.available
                ? "available"
                : "unavailable"}{" "}
            · FFmpeg:{" "}
            {typeof health.ffmpeg === "string"
              ? health.ffmpeg
              : health.ffmpeg.available
                ? "available"
                : "unavailable"}{" "}
            · Gemini:{" "}
            {typeof health.gemini === "string"
              ? health.gemini
              : health.gemini.configured
                ? health.gemini.available
                  ? "configured"
                  : "configured (unavailable)"
                : "not_configured"}{" "}
            · Diarization:{" "}
            {typeof health.diarization === "string"
              ? health.diarization
              : health.diarization.available
                ? "available"
                : "unavailable"}{" "}
            · TTS:{" "}
            {typeof health.tts === "string"
              ? health.tts
              : health.tts.available
                ? `available (${(health.tts.supported_languages || []).join(",")})`
                : "unavailable"}
          </p>
        )}
      </header>

      {phase !== 'completed' && (
        <section className="animate-rise mb-6 space-y-6 rounded-2xl border border-[rgba(15,61,62,0.12)] bg-white/75 p-6 shadow-[0_10px_40px_rgba(15,61,62,0.06)]">
          <LanguageSelector value={language} onChange={setLanguage} disabled={busy} />
          <AudioUploader
            file={file}
            durationLabel={localDuration}
            disabled={busy}
            onFileSelected={onFileSelected}
            onRemove={() => {
              setFile(null)
              setLocalDuration(null)
              setError(null)
            }}
          />
          <button
            type="button"
            disabled={!file || busy}
            onClick={handleProcess}
            className="w-full rounded-xl bg-[var(--coral)] px-5 py-3.5 text-base font-semibold text-white transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
          >
            Process Podcast
          </button>
        </section>
      )}

      {error && (
        <div
          className="mb-6 rounded-2xl border border-[rgba(163,59,43,0.3)] bg-[rgba(163,59,43,0.08)] px-5 py-4 text-[var(--err)]"
          role="alert"
        >
          {error}
        </div>
      )}

      {(phase === 'uploading' || phase === 'processing') && status && (
        <ProgressPanel
          stage={status.stage}
          stageLabel={status.stage_label}
          progress={status.progress}
          elapsedSeconds={status.elapsed_seconds}
        />
      )}

      {phase === 'completed' && result && <ResultsPanel result={result} onReset={resetAll} />}
    </div>
  )
}
