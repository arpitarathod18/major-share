# Meri Awaaz Hi Pehchan Hai — Backend

FastAPI backend for multilingual podcast processing.

## Windows setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and optionally set `GEMINI_API_KEY`.

## Run

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- API: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/api/health

## Tests

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pytest -q
```
