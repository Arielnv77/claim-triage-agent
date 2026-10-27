"""Knowledge base loading and ingestion of claims into Qdrant."""

import json
import os
import time
import uuid
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Folder (inside the container) with the historical claims in JSON
DATA_FOLDER = Path("/app/data/knowledge-base")
COLLECTION = "claims"
# Local embedding model: runs in the container, with no external API
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Created once and reused (loading the model is expensive)
_model = None
_client = None


def get_model() -> TextEmbedding:
    """Return the embedding model, creating it on first use."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def get_client() -> QdrantClient:
    """Return the Qdrant client, creating it on first use."""
    global _client
    if _client is None:
        _client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    return _client


def load_claims() -> list[dict]:
    """Read every JSON file in the knowledge base and return them as a list."""
    claims = []
    for path in sorted(DATA_FOLDER.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            claims.append(json.load(f))
    return claims


def wait_for_qdrant(attempts: int = 30, wait: float = 2.0) -> None:
    """Retry contacting Qdrant: on startup it may take a few seconds."""
    client = get_client()
    for attempt in range(1, attempts + 1):
        try:
            client.get_collections()
            return
        except Exception as error:
            print(f"Qdrant is not responding yet ({attempt}/{attempts}): {error}")
            time.sleep(wait)
    raise RuntimeError("Qdrant did not become available in time")


def _point_id(claim_id: str) -> str:
    """Stable ID derived from the claim_id: re-ingesting overwrites, never duplicates."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, claim_id))


def ingest() -> None:
    """Create the 'claims' collection if it does not exist and store the claims.

    It is idempotent: the IDs are derived from the claim_id, so running it
    twice overwrites the same points instead of duplicating them.
    """
    wait_for_qdrant()
    client = get_client()

    claims = load_claims()
    if not claims:
        print("There are no claims to ingest")
        return

    # If the collection is already complete, we save ourselves loading the model
    if (
        client.collection_exists(COLLECTION)
        and client.count(COLLECTION, exact=True).count >= len(claims)
    ):
        print(f"The collection '{COLLECTION}' is already populated; not re-ingesting")
        return

    descriptions = [claim["description"] for claim in claims]
    vectors = [vector.tolist() for vector in get_model().embed(descriptions)]

    # The collection is created with the model's actual vector size
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=len(vectors[0]), distance=Distance.COSINE
            ),
        )

    points = [
        PointStruct(
            id=_point_id(claim["claim_id"]), vector=vector, payload=claim
        )
        for claim, vector in zip(claims, vectors)
    ]
    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    print(f"Ingested {len(points)} claims into the collection '{COLLECTION}'")
