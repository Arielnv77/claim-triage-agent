"""API entry point."""

import os
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pypdf import PdfReader

from app.agent import run_triage
from app.intake import validate_intake
from app.knowledge import get_client, ingest
from app.safety import check_safety

# Env vars required to reach the LLM API; only their presence is checked
LLM_ENV_VARS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup, ingest the knowledge base into Qdrant."""
    ingest()
    yield


app = FastAPI(title="Claims API", lifespan=lifespan)

# The browser UI is served from another origin, so it needs explicit CORS access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health() -> dict:
    """Health endpoint: reports the state of the vector DB and the LLM config."""
    try:
        # A cheap round trip is enough to tell whether Qdrant is reachable
        get_client().get_collections()
        vector_db = "connected"
    except Exception:
        vector_db = "disconnected"

    # Presence only: the values are credentials and are never read out or logged
    llm = "configured" if all(os.environ.get(name) for name in LLM_ENV_VARS) else "not_configured"

    return {"status": "healthy", "services": {"vector_db": vector_db, "llm": llm}}


# In-memory store: claim_id -> {"response": TriageResponse, "trace": [...], "duration_ms": int}
# Enough for a single-process demo; a real deployment would use a database.
CLAIMS: dict[str, dict] = {}

# Upper bound on the document text handed to the model, in characters
MAX_DOCUMENT_CHARS = 20000


def extract_document_text(data: bytes) -> str:
    """Extract the text of a PDF upload; on failure log it and ignore the document."""
    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()[:MAX_DOCUMENT_CHARS]
    except Exception as exc:
        # A broken or non-PDF attachment must never fail the whole claim
        print(f"Document text extraction failed, ignoring the document: {exc!r}")
        return ""


@app.post("/api/v1/claims/triage")
async def triage_claim(
    description: str = Form(default=""),
    policy_type: str = Form(default="standard_homeowner"),
    # A single photo or several views of the same incident; both names are accepted
    image: Optional[UploadFile] = File(default=None),
    images: Optional[list[UploadFile]] = File(default=None),
    # Optional PDF with extra evidence about the same incident
    document: Optional[UploadFile] = File(default=None),
) -> dict:
    """Run the triage agent over a claim and store the report for later retrieval."""
    uploads = ([image] if image is not None else []) + list(images or [])
    image_bytes = [await upload.read() for upload in uploads]
    image_bytes = [data for data in image_bytes if data]

    # The PDF is evidence only; an unreadable one leaves the text empty
    document_bytes = await document.read() if document is not None else b""
    document_text = extract_document_text(document_bytes) if document_bytes else ""

    if not image_bytes and not description.strip():
        # Nothing to work with: neither a photo nor any text.
        # Returned directly (not via HTTPException) so the keys stay top-level
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_input",
                "message": "Provide at least an image or a non-empty description.",
            },
        )

    # Intake guardrails run before the agent so rejected claims cost nothing
    intake_validation = validate_intake(description, image_bytes, document_text)
    if any(value is False for value in intake_validation.values()):
        # Returned directly (not via HTTPException) so the keys stay top-level
        return JSONResponse(
            status_code=400,
            content={
                "error": "intake_rejected",
                "intake_validation": intake_validation,
                "message": "The claim was rejected by intake validation.",
            },
        )

    started = time.monotonic()
    result = run_triage(
        description=description,
        policy_type=policy_type,
        images=image_bytes,
        document_text=document_text,
    )

    # Output guardrails: may rewrite the report's text fields in place
    safety_checks = check_safety(result["report"])
    duration_ms = int((time.monotonic() - started) * 1000)

    if any(value is False for value in safety_checks.values()):
        # Unsafe output is never returned, not even partially
        return JSONResponse(
            status_code=422,
            content={
                "error": "safety_check_failed",
                "safety_checks": safety_checks,
                "message": "The generated report failed the output safety checks.",
            },
        )

    claim_id = str(uuid.uuid4())
    response = {
        "claim_id": claim_id,
        "intake_validation": intake_validation,
        "triage_assessment": result["report"],
        "safety_checks": safety_checks,
    }
    CLAIMS[claim_id] = {
        "response": response,
        "trace": result["trace"],
        "duration_ms": duration_ms,
    }
    return response


@app.get("/api/v1/claims/{claim_id}")
def get_claim(claim_id: str) -> dict:
    """Return a previously computed triage report."""
    stored = CLAIMS.get(claim_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Claim not found")
    return stored["response"]


@app.get("/api/v1/agent/trace/{claim_id}")
def get_trace(claim_id: str) -> dict:
    """Return the ReAct trace of a triage run, numbering the steps in order."""
    stored = CLAIMS.get(claim_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    steps = [
        {"step_number": number, **step}
        for number, step in enumerate(stored["trace"], start=1)
    ]
    return {
        "claim_id": claim_id,
        "steps": steps,
        "total_duration_ms": stored["duration_ms"],
    }
