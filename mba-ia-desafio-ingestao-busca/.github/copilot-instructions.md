# Copilot / AI Agent Instructions

Quick reference to become productive in this repo.

## Big picture
- Purpose: Ingest a PDF, index it with embeddings (pgVector on Postgres) and answer user questions using only retrieved text.
- Main flow:
  1. `src/ingest.py` loads `document.pdf` (repo root), splits text into chunks (1000 chars, 150 overlap), computes embeddings and writes vectors to Postgres via `PGVector.from_documents`.
  2. `src/search.py` exposes `search_top_k(query, k)` which uses `PGVector.similarity_search_with_score` and `format_context(results, max_chars)` to build prompt context.
  3. `src/chat.py` builds a strict prompt that **must** reply only from the retrieved context (fallback string used when context is insufficient) and calls the configured LLM.

## Where to look (key files)
- `src/ingest.py` — PDF loading (`PyPDFLoader`), text splitting (`RecursiveCharacterTextSplitter`), chunking constants (`DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP`), ingestion via `PGVector`.
- `src/search.py` — vectorstore instantiation (`PGVector`), `search_top_k`, `format_context` (joins page contents with `\n\n` and truncates by slicing).
- `src/chat.py` — prompt rules (strict fallback), LLM selection (`ChatOpenAI` / `ChatGoogleGenerativeAI`), `answer_question` and CLI loop.
- `README.md` and `.env.example` — usage steps and environment variables.

## Environment & run commands (essential)
- Python 3.10+; install deps: `pip install -r requirements.txt`.
- Configure `.env` from `.env.example` (set `PROVIDER`, API keys and Postgres credentials).
- Start Postgres (pgVector-enabled): `docker compose up -d`.
- Ingest: `python src/ingest.py` (expects `document.pdf` at repo root).
- Run chat (CLI): `python src/chat.py`.

## Important conventions & gotchas
- Provider switch: set `PROVIDER=openai` or `PROVIDER=gemini`. The code chooses embeddings and LLM automatically using `OPENAI_*` or `GEMINI_*` env vars.
- Fallback string: `"Não tenho informações necessárias para responder sua pergunta."` — prompt logic depends on exact wording; avoid changing unless you update all checks.
- Temperature is set to `0` in `chat.py` to reduce generation variance — preserve this for deterministic behavior.
- `format_context` truncates via simple slicing (`context[:max_chars]`) — contexts may be cut mid-sentence; when adding features, consider a sentence-aware truncation.
- `PG_COLLECTION` controls the Postgres collection/table name; change via `.env`.
- Ingestion requires at least one PDF in repo root; `ingest.py` raises if none found.
- Logging: modules use `logging` at INFO level by default; change to DEBUG for verbose troubleshooting.

## Quick code examples (use exact symbols)
- Search & format:
```py
from search import search_top_k, format_context
results = search_top_k("Minha pergunta", k=5)
context = format_context(results, max_chars=6000)
```
- Ingest: ensure `.env` and `document.pdf` exist then:
```bash
python src/ingest.py
```

## Integration points
- Postgres with pgVector: vectors are stored in a PG collection; ensure the DB is reachable using `PG_*` env values.
- OpenAI / Google Gemini: embeddings and chat are provided by the SDK wrappers used in code (`langchain_openai`, `langchain_google_genai`).

## What the agent should avoid doing
- Do not answer questions using world knowledge—responses must be built from retrieved context only.
- Avoid changing the fallback wording or the prompt semantics without updating `chat.py` tests/examples and README.

---

If any section is unclear or you'd like more examples (e.g., how to add page/file metadata to the `PyPDFLoader`), tell me which part to expand and I will iterate. ✅