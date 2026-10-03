import axios, { AxiosError } from 'axios'
import type {
  HealthResponse,
  LanguageCode,
  ProcessResponse,
  ProcessResult,
  StatusResponse,
  UploadResponse,
} from '../types'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000',
  timeout: 120000,
})

export function getErrorMessage(err: unknown, fallback = 'Something went wrong.'): string {
  if (axios.isAxiosError(err)) {
    const ax = err as AxiosError<{ error?: string; detail?: string }>
    const data = ax.response?.data
    if (data && typeof data === 'object') {
      if (typeof data.error === 'string' && data.error.trim()) return data.error
      if (typeof data.detail === 'string' && data.detail.trim()) return data.detail
    }
    if (ax.code === 'ERR_NETWORK') {
      return 'Cannot reach the backend. Make sure it is running at http://127.0.0.1:8000'
    }
    return ax.message || fallback
  }
  if (err instanceof Error) return err.message
  return fallback
}

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/api/health')
  return data
}

export async function uploadAudio(file: File): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<UploadResponse>('/api/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function startProcessing(
  uploadId: string,
  outputLanguage: LanguageCode,
): Promise<ProcessResponse> {
  const { data } = await api.post<ProcessResponse>('/api/process', {
    upload_id: uploadId,
    output_language: outputLanguage,
  })
  return data
}

export async function getStatus(jobId: string): Promise<StatusResponse> {
  const { data } = await api.get<StatusResponse>(`/api/status/${jobId}`)
  return data
}

export async function getResult(jobId: string): Promise<ProcessResult> {
  const { data } = await api.get<ProcessResult>(`/api/result/${jobId}`)
  return data
}

export function downloadUrl(path: string): string {
  const base = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
  return `${base}${path}`
}

export { api }
