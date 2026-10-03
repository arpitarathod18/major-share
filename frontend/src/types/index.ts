export type LanguageCode = 'en' | 'hi' | 'kn' | 'te'

export interface UploadResponse {
  success: boolean
  upload_id: string
  filename: string
  size_bytes: number
  duration_seconds: number | null
  duration_formatted: string | null
  error?: string
  stage?: string
}

export interface ProcessResponse {
  success: boolean
  job_id: string
  status: string
  error?: string
  stage?: string
}

export interface StatusResponse {
  success: boolean
  job_id: string
  status: 'queued' | 'processing' | 'completed' | 'failed' | string
  stage: string
  stage_label: string
  progress: number
  elapsed_seconds: number | null
  error?: string | null
  error_stage?: string | null
  warnings?: string[]
}

export interface TranscriptWord {
  start: number
  end: number
  text: string
}

export interface TranscriptSegment {
  start: number
  end: number
  duration: number
  text: string
  speaker: string | null
  language: string
  language_name: string
  timestamp: string
  words?: TranscriptWord[]
}

export interface ProcessResult {
  success: boolean
  job_id: string
  filename: string
  duration_seconds: number
  duration_formatted: string
  detected_language: string
  detected_language_name: string
  detected_language_probability: number
  detected_confident: boolean
  detected_language_detail?: {
    code: string
    name: string
    confidence: number
    alternatives: Record<string, number>
    sampled_regions: number
    code_switched: boolean
  }
  speakers?: string[]
  detected_speakers?: number
  single_speaker?: boolean
  original_audio_url?: string
  output_language: string
  output_language_name: string
  processing_time_seconds: number
  transcript: {
    full_text: string
    diarized: boolean
    speaker_count: number
    detected_speakers?: number
    single_speaker?: boolean
    message: string | null
    segments: TranscriptSegment[]
    formatted: string
    method?: string
    overlap_supported?: boolean
  }
  validation?: {
    passed: boolean
    warnings: string[]
    errors: string[]
    details: Record<string, any>
  }
  summary: {
    text: string
    method: string
    language: string
  }
  audio_summary: {
    available: boolean
    method: string | null
    filename: string | null
    url: string | null
    message: string | null
  }
  downloads: {
    transcript: string
    summary: string
    audio: string | null
  }
  warnings: string[]
}

export interface ApiError {
  success: false
  error: string
  stage?: string
}

export interface HealthResponse {
  backend: string
  whisper: string | { available: boolean; model?: string }
  ffmpeg: string | { available: boolean; detail?: string }
  gemini: string | { configured: boolean; available: boolean; model?: string }
  diarization: string | { available: boolean }
  tts: string | { available: boolean; supported_languages?: string[] }
}
