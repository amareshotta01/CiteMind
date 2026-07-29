# CiteMind

A trust-first, hybrid-retrieval document Q&A system — fixing what NotebookLM gets wrong.

## Phase A — current status
Streamlit UI ↔ FastAPI backend ↔ MongoDB, with mocked RAG logic (real
retrieval/citations come in later phases). This proves the plumbing works
end-to-end before the harder logic is built.

## Setup

1. Create a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install MongoDB locally (or use MongoDB Atlas free tier):
   - Local: https://www.mongodb.com/try/download/community
   - Atlas (no local install needed): https://www.mongodb.com/cloud/atlas/register

4. Copy `.env.example` to `.env` and fill in values:
   ```
   cp .env.example .env
   ```

## Running (two terminals needed)

**Terminal 1 — backend:**
```
uvicorn backend.main:app --reload --port 8000
```
Check it's alive at http://localhost:8000/health

**Terminal 2 — frontend:**
```
streamlit run ui/streamlit_app.py
```

## What's mocked right now
- `/upload` accepts a file but doesn't chunk/embed it yet
- `/query` returns a fixed fake answer, not a real retrieved one

These get replaced with real logic in the next phases — the request/response
shape stays the same, so the UI won't need to change.

## Project structure
See `CiteMind_Project_Spec.md` for full architecture and roadmap.
