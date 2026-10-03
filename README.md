# Meri Awaaz Hi Pehchan Hai

**Multilingual Podcast Processing System for India**

A localhost web application that accepts podcast/audio files and runs:

Audio Upload → Language Detection → Speech-to-Text → Speaker Diarization → Multilingual Summarization → Text-to-Speech Audio Summary

Supported languages: **English**, **Hindi**, **Kannada**, **Telugu**.

---

## Requirements (Windows)

1. **Python 3.10+** (tested with 3.11)
2. **Node.js 18+** (tested with 22)
3. **FFmpeg** on PATH (`ffmpeg` and `ffprobe`)
4. Internet access for first Whisper model download and TTS (edge-tts / gTTS)
5. Optional: **Gemini API key** for higher-quality multilingual summaries/translation

---

## 1. Install FFmpeg (if needed)

1. Download a full build from https://www.gyan.dev/ffmpeg/builds/
2. Extract and add the `bin` folder to your Windows PATH
3. Verify in PowerShell:

```powershell
ffmpeg -version
ffprobe -version
```

---

## 2. Backend setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Edit `backend\.env`:

```
GEMINI_API_KEY=
WHISPER_MODEL=medium
DEVICE=cpu
MAX_AUDIO_DURATION=3600
```

Set `GEMINI_API_KEY` if you have one. The app still runs without it (extractive summary fallback).

### Start backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- API: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/api/health

---

## 3. Frontend setup

Open a **second** PowerShell window:

```powershell
cd frontend
npm install
npm run dev
```

- App: http://localhost:5173

---

## 4. How to use

1. Open http://localhost:5173
2. Choose output language (English / Hindi / Kannada / Telugu)
3. Upload an MP3, WAV, or M4A podcast file
4. Click **Process Podcast**
5. Wait for stages to complete
6. View transcript, summary, and audio summary
7. Download results

---

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Component status |
| POST | `/api/upload` | Upload audio |
| POST | `/api/process` | Start async job |
| GET | `/api/status/{job_id}` | Poll progress |
| GET | `/api/result/{job_id}` | Final result |
| GET | `/api/download/{job_id}/transcript` | Download transcript |
| GET | `/api/download/{job_id}/summary` | Download summary |
| GET | `/api/download/{job_id}/audio` | Download audio summary |

---

## Backend tests

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pytest -q
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| FFmpeg errors | Install FFmpeg full build and restart the terminal/backend |
| Backend unreachable in UI | Start uvicorn on port 8000 first |
| Empty transcript | Use clearer speech / louder audio / shorter clip |
| Gemini warnings | Set `GEMINI_API_KEY` in `backend/.env` |
| Slow first run | Whisper downloads the `medium` model once |
| TTS missing | Check internet; edge-tts/gTTS need network |
| CUDA issues | Keep `DEVICE=cpu` (default) |

---

## Project structure

```
Major/
├── backend/          FastAPI + AI services
├── frontend/         React + Vite + Tailwind
├── README.md
└── .gitignore
```

---

## Notes

- Processing is **job-based** (non-blocking). The UI polls `/api/status/{job_id}`.
- Speaker diarization uses real audio embeddings + clustering. If it fails, a normal transcript is returned (no fake speaker labels).
- Summarization prefers Gemini; otherwise uses a local extractive fallback.
- TTS prefers **edge-tts** (en/hi/kn/te), then **gTTS**.
