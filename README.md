# Enterprise Document RAG Assistant

A retrieval-augmented generation (RAG) service for enterprise PDFs. Upload a
document, ask questions, and get an answer grounded in the retrieved passages
with page-level citations — and a clear refusal when the document does not
contain the answer.

Everything runs locally: PyMuPDF for parsing, `sentence-transformers` for
embeddings, ChromaDB for the vector store, and any OpenAI-compatible chat model
for generation. No document text leaves the host except in the LLM request you
opt into.

---

## Table of contents

- [Status and what is verified](#status-and-what-is-verified)
- [Quick start](#quick-start)
- [Docker](#docker)
- [API](#api)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [How answering works](#how-answering-works)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Security notes](#security-notes)
- [Limitations](#limitations)
- [Project layout](#project-layout)

---

## Status and what is verified

Everything in this README was executed on this machine. Stated plainly, so the
numbers are not over-read:

| Area | Status |
| --- | --- |
| PDF ingestion, cleaning, chunking | Verified end to end on the sample document |
| Retrieval quality | Verified: hit@4 = 1.00, MRR = 0.95 on 13 labelled questions |
| Answer quality with the **non-LLM baseline** | Verified: keyword recall 0.89 |
| Answer quality with **OpenAI** | **Not verified — no API key was available** |
| API endpoints, error mapping, upload validation | Verified live over HTTP with `curl` |
| Test suite | 247 tests passing, 93% line coverage |
| Docker image | **Not built — no Docker daemon in this environment** |

The OpenAI client is implemented, unit-tested against a stubbed SDK (request
shape, error mapping, empty completions, timeouts) and wired end to end, but no
live call to a real model was made. Treat the generation-quality numbers below
as belonging to the offline baseline only.

---

## Quick start

Requires Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the example configuration and, if you want real generated answers, add a key:

```bash
cp .env.example .env
# edit .env: set OPENAI_API_KEY, or leave it unset to use the offline baseline
```

Start the API:

```bash
uvicorn app.main:app --reload
```

The service is then on <http://localhost:8000> with interactive docs at `/docs`.

Index the sample document and ask it something:

```bash
curl -F "file=@data/sample/northwind_lending_policy.pdf" localhost:8000/ingest

curl -H 'Content-Type: application/json' \
     -d '{"question":"What is the maximum loan tenure?"}' \
     localhost:8000/ask
```

```json
{
  "answer": "- Maximum loan tenure: 20 years.",
  "sources": [
    {
      "document": "northwind_lending_policy.pdf",
      "page": 2,
      "score": 0.5726,
      "chunk_id": "northwind-lending-policy-pdf-p0002-c0000-3cb33e5a47c2"
    },
    {
      "document": "northwind_lending_policy.pdf",
      "page": 3,
      "score": 0.4744,
      "chunk_id": "northwind-lending-policy-pdf-p0003-c0000-74cca98a4ae3"
    }
  ],
  "no_answer": false,
  "chunks_used": 4,
  "model": "extractive-baseline (non-LLM)",
  "top_score": 0.5726
}
```

That is the real response captured from a running server, with the two best
sources shown. It was produced by the offline extractive baseline, so the answer
is a sentence copied from page 2 — with `LLM_PROVIDER=openai` the same request
returns a model-written answer with the same shape. Note the two fields worth
reading: `top_score` is the best similarity found, and `no_answer` says whether
the model claimed the context was insufficient.

The first request after startup is slow (~45 s here) because the embedding model
is loaded lazily on first use. Later requests are sub-second.

To index without starting the API:

```bash
python scripts/ingest.py data/sample/northwind_lending_policy.pdf
python scripts/ingest.py --list
python scripts/ingest.py data/sample/*.pdf --reset
```

---

## Docker

```bash
docker compose up --build
```

Then:

```bash
curl localhost:8000/health
curl -F "file=@data/sample/northwind_lending_policy.pdf" localhost:8000/ingest
curl -H 'Content-Type: application/json' \
     -d '{"question":"What is the maximum loan tenure?"}' \
     localhost:8000/ask
```

The image is a two-stage build on `python:3.12-slim`: dependencies are compiled
in the builder stage and copied into a runtime image that has no compiler. The
container runs as a non-root user, stores the vector store and uploads on the
`rag-data` volume so they survive rebuilds, and ships a `/health` healthcheck.

The `Dockerfile` and `docker-compose.yml` are written and their configuration
was validated against the real settings schema, but **the image was not built
here** because this machine has no Docker daemon. Expect to shake out build
issues on first run.

Override configuration through the environment, for example
`LLM_PROVIDER=openai OPENAI_API_KEY=sk-... docker compose up`.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness plus vector-store and LLM readiness |
| `POST` | `/ingest` | Upload and index a PDF (`multipart/form-data`, field `file`) |
| `POST` | `/ask` | Ask a question about indexed documents |
| `GET` | `/documents` | List indexed documents and chunk count |
| `GET` | `/docs`, `/redoc` | Interactive API documentation |

### `GET /health`

Never loads the embedding model, so it is safe to poll. Returns `200` with
`"status": "ok"` when ready and `503` with `"status": "degraded"` when the
vector store is unreachable or the LLM provider is not configured.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "local",
  "vector_store": {"status": "ok", "chunks": 7},
  "llm": {"provider": "openai", "model": "gpt-4o-mini", "configured": true}
}
```

### `POST /ingest`

```bash
curl -F "file=@document.pdf" localhost:8000/ingest
```

```json
{
  "message": "Document ingested successfully",
  "document": "northwind_lending_policy.pdf",
  "pages_processed": 7,
  "chunks_created": 7,
  "chunks_replaced": 0,
  "total_words": 754,
  "average_chunk_words": 107.7,
  "duration_ms": 384
}
```

Re-uploading the same file replaces its previous chunks, so ingestion is
idempotent and the collection does not grow without bound.

### `POST /ask`

```json
{
  "question": "What is the maximum loan tenure?",
  "top_k": 4
}
```

`top_k` is optional and bounded (1–20). The response contains the answer, one
entry per cited source with its page and similarity score, a `no_answer` flag
and the top similarity.

### Errors

Every error uses one envelope, and internal details are never returned to the
caller:

```json
{"error": {"code": "invalid_pdf", "message": "The uploaded file does not look like a PDF (missing %PDF- header)."}}
```

| Status | `code` | Cause |
| --- | --- | --- |
| `400` | `invalid_pdf` | Not a PDF, or unreadable |
| `404` | `http_error` | Unknown route |
| `413` | `payload_too_large` | Upload exceeds `MAX_UPLOAD_SIZE_MB` |
| `415` | `unsupported_file_type` | Extension not allowed |
| `422` | `empty_document` | No extractable text (for example a scanned PDF) |
| `422` | `validation_error` | Request body failed validation |
| `500` | `retrieval_failed` | Vector store search failed |
| `502` | `llm_provider_error` | The LLM provider failed or timed out |
| `503` | `llm_not_configured` | `OPENAI_API_KEY` missing while `LLM_PROVIDER=openai` |
| `503` | `vector_store_unavailable` | Vector store unreachable |

These responses were each confirmed against a running server.

---

## Configuration

All settings are read from the environment or a `.env` file; see
[`.env.example`](.env.example) for the annotated list. The ones that matter
most:

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai` or `extractive` (offline baseline) |
| `OPENAI_API_KEY` | — | Required for `openai`; stored as a `SecretStr`, never logged |
| `OPENAI_BASE_URL` | — | Point at any OpenAI-compatible gateway |
| `LLM_MODEL` | `gpt-4o-mini` | Any chat model |
| `LLM_TEMPERATURE` | `0.0` | Zero for reproducible answers |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | 384 dimensions |
| `EMBEDDING_DEVICE` | auto | Set to `cpu` to force CPU inference |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `600` / `80` | **Counted in words, not model tokens** |
| `TOP_K` | `4` | Chunks retrieved per question |
| `SIMILARITY_THRESHOLD` | unset | Optional minimum cosine similarity |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db` | Vector store location |
| `MAX_UPLOAD_SIZE_MB` | `25` | Upload ceiling |
| `LOG_LEVEL` | `INFO` | |

Invalid combinations fail at startup rather than mid-request: for example
`CHUNK_OVERLAP >= CHUNK_SIZE` and an unsupported `LOG_LEVEL` are both rejected
by settings validation.

### The extractive baseline

`LLM_PROVIDER=extractive` swaps the OpenAI client for a deterministic,
non-LLM answering strategy that selects the context sentences with the highest
lexical overlap with the question and stitches them together with citations.

It exists so the whole pipeline can be exercised, tested and demonstrated
without an API key or network access. **It is not a language model.** Its
answers are copied sentences, not generated prose, and it has no real notion of
"the document does not answer this" — which is why its abstention behaviour is
weak. Never present its output as model quality.

---

## Architecture

```
        PDF bytes
            │
            ▼
   ┌─────────────────┐   per-page text + page numbers + file hash
   │   PdfLoader     │   (PyMuPDF; rejects files with no text layer)
   └────────┬────────┘
            ▼
   ┌─────────────────┐   unicode/whitespace repair, de-hyphenation,
   │   TextCleaner   │   repeated header/footer removal
   └────────┬────────┘
            ▼
   ┌─────────────────┐   sentence-aware word windows with overlap,
   │     Chunker     │   never spanning a page boundary
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │    Embedder     │   sentence-transformers, lazy, L2-normalised
   └────────┬────────┘
            ▼
   ┌─────────────────┐   upsert, delete-by-document, cosine search
   │  ChromaVector   │   (persistent on disk)
   │     Store       │
   └────────┬────────┘
            ▼
   ┌─────────────────┐   query embedding, top-k, optional threshold
   │   Retriever     │
   └────────┬────────┘
            ▼
   ┌─────────────────┐   grounded prompt with [S1], [S2] labels
   │  RAGPipeline    │   → LLM or extractive baseline
   └────────┬────────┘
            ▼
      answer + sources + no_answer
```

Design decisions worth calling out:

- **Chunks never cross a page boundary.** Citations stay exact, and a reader can
  always open the right page.
- **Chunk IDs are deterministic** (document, page, position, content hash), and
  ingestion deletes a document's prior chunks before writing. Re-ingesting the
  same file is a no-op instead of a duplicate.
- **Heavy dependencies are lazy.** The embedding model and the LLM client are
  constructed on first use, so importing the app, running `--help` and serving
  `/health` never pay the model load.
- **Errors are a hierarchy** (`InvalidPdfError`, `EmptyDocumentError`,
  `LLMProviderError`, …) each carrying its own HTTP status and stable error code.
  The API layer maps them centrally.
- **The vector store and the LLM are protocols**, so tests inject doubles and the
  store could be swapped for a hosted service without touching the pipeline.

---

## How answering works

1. The question is embedded and matched against the collection.
2. The top `top_k` chunks above the optional threshold become the context.
3. If the context is empty, the request is answered immediately with a refusal
   — **the LLM is never called**, so an unanswerable question costs nothing and
   cannot hallucinate.
4. Otherwise the context is wrapped in labelled blocks (`[S1]`, `[S2]`, …) and
   the system prompt instructs the model to use only that context, to say when
   the information is absent, and to cite the labels it used.
5. The response is parsed for those labels and matched back to the chunks, which
   is what populates `sources` and the `model`-reported `no_answer` flag.

Citations are derived from what the model actually referenced, not from what was
retrieved, so the sources list reflects the answer.

---

## Evaluation

`evaluation/questions.json` holds 16 hand-written questions against the sample
document: 13 answerable with an expected page and expected keywords, and 3 that
the document deliberately does not answer.

```bash
python scripts/ingest.py data/sample/northwind_lending_policy.pdf

# retrieval only: no LLM, no API key
python scripts/evaluate.py --retrieval-only

# full run, machine-readable report
python scripts/evaluate.py --json-out evaluation/report.json
```

Results on this machine, with the real embedding model and ChromaDB:

| Metric | Value |
| --- | --- |
| Retrieval hit@4 | **1.00** |
| MRR | **0.95** |
| Top-1 retrieval | **0.92** |
| Source-page match | **1.00** |
| Keyword recall (extractive baseline) | **0.89** |
| All keywords present (extractive baseline) | **0.85** |
| Abstention (extractive baseline) | **0.67** |

Read these honestly:

- **Retrieval is the strong part.** Every answerable question retrieved its
  expected page within the top 4, and 12 of 13 at rank 1.
- **The answer metrics describe the offline baseline, not a model.** They are
  produced by the extractive strategy, which copies sentences.
- **Abstention is 0.67 and that is a baseline limitation.** It refused 2 of 3
  unanswerable questions; on `q14` ("how many retail branches…") it returned a
  document-control block that happened to contain the words "retail branches".
  A real LLM asked to say when context is insufficient does this far better.
  This metric is only meaningful once run against `LLM_PROVIDER=openai`.
- **Two questions score low even so**: `q05` (required document list, recall
  0.00) and `q10` (insurance, 0.60). Both are sentence-selection limits of the
  baseline; retrieval found the right pages in both cases.
- There is **no LLM judge and no human annotation**, and 16 questions is far too
  small a sample for a statistically meaningful claim. This is a regression
  baseline for catching breakage, not a benchmark.

`tests/test_sample_document.py` asserts that every expected keyword still
appears on its expected page, so the question set cannot silently drift away
from the sample document.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest                                    # whole suite
pytest --cov=app --cov=scripts --cov-report=term-missing
pytest tests/test_rag.py -v               # one module
```

247 tests, 93% line coverage, in roughly two to five minutes (the real-model
tests dominate the runtime).

| Module | What it covers |
| --- | --- |
| `test_chunker.py` | Window boundaries, overlap, page isolation, determinism |
| `test_ingestion.py` | Loading, cleaning, marginalia removal, chunk metadata |
| `test_retrieval.py` | Ranking, thresholds, empty results, store failures |
| `test_rag.py` | Prompting, citation parsing, refusals, ingestion idempotency |
| `test_llm.py` | OpenAI client against a stubbed SDK: request shape, error mapping, empty completions, no key leakage |
| `test_api.py` | Every endpoint, status code and error envelope |
| `test_main.py` | App factory, lifespan, OpenAPI, config reporting |
| `test_scripts.py` | Evaluation scoring, question-set validation, both CLIs |
| `test_sample_document.py` | Sample PDF generation and its agreement with the eval set |
| `test_integration_real_model.py` | The real embedding model end to end (auto-skips if unavailable) |

Unit tests use deterministic doubles for the embedder, LLM and vector store, so
they run in milliseconds and never touch the network. Only
`test_integration_real_model.py` loads the actual model; it skips cleanly when
the model files are not present.

---

## Security notes

Implemented:

- Uploads are validated by extension, MIME type, size, and the `%PDF-` magic
  bytes — a renamed executable is rejected.
- Filenames are reduced to a bare, sanitised name, so `../../etc/passwd` cannot
  escape the upload directory.
- `OPENAI_API_KEY` is a `SecretStr`: it is not logged, not returned by any
  endpoint, and excluded from the startup configuration dump.
- Error responses are generic; the underlying exception goes to the log, not to
  the client.
- The Docker image runs as a non-root user.

Deliberately **not** implemented — this is a single-tenant reference service:

- No authentication or authorization on any endpoint. Anyone who can reach the
  port can ingest and query. Put it behind an authenticating gateway before
  exposing it.
- No rate limiting, no per-tenant isolation, one shared collection.
- Uploaded PDFs are written to disk unencrypted and are not deleted after
  ingestion.
- No TLS, no audit log, no secrets manager integration.

---

## Limitations

- **Text-layer PDFs only.** A scanned PDF is rejected with `422 empty_document`.
  OCR is out of scope; add it before ingesting scanned archives.
- **Dense retrieval only.** No hybrid keyword/BM25 search, no reranking, no
  query rewriting, and one vector store for all documents. Retrieval was already
  the strongest component on the sample set, but a real corpus with many
  near-duplicate policies would want hybrid search and filters.
- **Chunking is word-count based**, not token-based, so a chunk's real token cost
  depends on the document's vocabulary.
- **No table or layout extraction.** Tabular data is read as linear text, which
  will mangle complex financial tables.
- **Header/footer removal is heuristic**, tuned for repeated edge lines. A
  document with a unique footer per page keeps that text.
- **ChromaDB is single-node on local disk.** Fine for one process; not a
  production multi-tenant store. The `VectorStore` protocol exists to allow a
  swap.
- **One model per process.** Each Uvicorn worker loads its own copy of the
  embedding model, which is why the container defaults to a single worker.
- **No conversation memory.** Each question is independent; multi-turn follow-ups
  such as "and the fee?" have no referent.
- **No streaming, no citations rendered into a PDF, no export formats.**

---

## Project layout

```
app/
  main.py              FastAPI factory, lifespan, exception handlers
  core/                settings, error hierarchy, logging
  ingestion/           pdf_loader, cleaner, chunker, models
  embeddings/          Embedder protocol + sentence-transformers implementation
  vectorstore/         VectorStore protocol + ChromaDB implementation
  retrieval/           Retriever and result models
  llm/                 LLM protocol, OpenAI client, extractive baseline
  rag/                 pipeline, prompts, citation parsing
  api/                 routes, request/response schemas, upload validation
scripts/
  ingest.py            index PDFs from the command line
  evaluate.py          run the labelled evaluation and report metrics
  make_sample_pdf.py   regenerate the synthetic sample document
evaluation/
  questions.json       16 labelled questions
data/sample/           synthetic 7-page demo document
tests/                 247 tests
```

## License

MIT — see [LICENSE](LICENSE). The sample document is fictional: "Northwind
Capital" is an invented company and its policy states no real product,
regulation or customer.
