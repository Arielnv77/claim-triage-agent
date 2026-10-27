"""Semantic search over the historical claims stored in Qdrant."""

from app.knowledge import COLLECTION, get_client, get_model


def search_past_claims(query: str, limit: int = 3) -> list[dict]:
    """Return the past claims most similar to the query."""
    # The query is vectorized with the same model used at ingestion time
    vector = next(iter(get_model().embed([query]))).tolist()

    results = get_client().query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=limit,
        with_payload=True,
    ).points

    claims = []
    for point in results:
        payload = point.payload or {}
        claims.append(
            {
                "claim_id": payload.get("claim_id"),
                "similarity_score": point.score,
                "resolution_summary": (
                    f"{payload.get('resolution')} - "
                    f"actual repair cost: ${payload.get('repair_cost_actual')}"
                ),
            }
        )
    return claims
