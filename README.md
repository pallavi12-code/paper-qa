# Scientific Paper QA

Upload PDFs, ask questions, get answers with page citations.
Stack: Streamlit · PyMuPDF · Sentence Transformers · FAISS · Claude or Gemini.

## Setup (Mac)

```bash
cd paper-qa
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
open -e .env        # add your API key
```

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

Opens at http://localhost:8501. Upload PDFs in the sidebar, then ask away.
First run downloads the embedding model (~90 MB).

## Test

```bash
python -m pytest -q
```

Tests use a fake embedder and fake LLM, so they need no API key or model download.

## Config (.env)

| Var | Meaning |
|---|---|
| `LLM_PROVIDER` | `anthropic` or `gemini` |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Anthropic settings |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Gemini settings |

## Layout

```
app.py              Streamlit UI
sciqa/pdf_loader.py PDF -> page-tagged chunks
sciqa/index.py      embeddings + FAISS search
sciqa/qa.py         prompt + answer with citations
sciqa/llm.py        LLM provider call
tests/              pytest suite
```

## Limits (v1)

- Text-based PDFs only (no OCR for scanned papers).
- Index lives in memory; re-uploading rebuilds it.
- arXiv / PubMed ingestion: planned next.
