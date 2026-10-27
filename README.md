# Claim Triage Agent

An AI agent that triages property insurance claims. Submit a description, photos and an optional PDF; the agent searches similar past claims, estimates the repair cost, checks policy coverage and returns a structured report together with its full reasoning trace.

## Features

- **Hand-written ReAct loop** (no agent framework) with three tools: semantic search over past claims, a repair cost estimator and a coverage lookup.
- **RAG** over historical claims with Qdrant and local embeddings (`BAAI/bge-small-en-v1.5` via fastembed, no external embedding API).
- **Multimodal input**: one or more JPEG/PNG photos and an optional PDF document as evidence.
- **Intake guardrails**: regex PII detection plus an LLM check for topic, tone and image usability.
- **Output guardrails**: an LLM safety review (no coverage promises, legal advice, settlement amounts, fault assignment or emotive wording) with one automatic rewrite, failing closed.
- **Traceability**: every reason / act / observe step is stored and exposed through the API and the UI.
- **Prompt-injection hardening**: documents and photos are treated as evidence only; instructions inside them are ignored.

## Architecture

```
ui (nginx, :3000)  ──►  api (FastAPI, :8000)  ──►  LLM (any OpenAI-compatible API)
                              │
                              └──►  qdrant (vector DB, :6333)
```

| Path | Purpose |
| --- | --- |
| `api/app/agent.py` | ReAct loop, tool schemas and system prompt |
| `api/app/intake.py` | Input validation and PII detection |
| `api/app/safety.py` | Output safety checks and rewrite |
| `api/app/knowledge.py`, `search.py` | Claim ingestion and semantic search |
| `api/app/tools.py` | Deterministic cost and coverage lookups |
| `ui/index.html` | Single-file web UI |
| `data/`, `tools/` | **Synthetic** sample claims and lookup tables |

## Quick start

Requirements: Docker and an API key for an OpenAI-compatible chat API that supports tool calling and image input.

```bash
cp .env.example .env    # then fill in your key, base URL and model
make setup              # build and start qdrant, api and ui
```

Open http://localhost:3000.

```bash
make test   # offline unit tests, no LLM calls
make logs
make stop
```

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Vector DB and LLM configuration status |
| `POST` | `/api/v1/claims/triage` | Multipart form: `description`, `policy_type`, `images`, `document` |
| `GET` | `/api/v1/claims/{claim_id}` | Stored triage report |
| `GET` | `/api/v1/agent/trace/{claim_id}` | Step-by-step agent trace |

## Notes

- All data in `data/` and `tools/` is synthetic and only meant for demonstration.
- Reports are kept in memory; a real deployment would use a database.

## License

MIT, see [LICENSE](LICENSE).
