"""Query endpoint for RAG retrieval."""

from typing import Optional

from fastapi import APIRouter, Depends

from src.api.dependencies import get_retriever, get_tracing_service
from src.models.schemas import QueryRequest, QueryResponse
from src.observability.tracing import TracingService
from src.retrieval.retriever import Retriever

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    tracing: Optional[TracingService] = Depends(get_tracing_service),
) -> QueryResponse:
    """Query the RAG system and retrieve relevant document chunks."""
    if tracing is None:
        return await retriever.retrieve(request)

    outputs: dict = {}
    async with tracing.trace_query(
        query=request.query,
        retrieval_mode=request.retrieval_mode,
        top_k=request.top_k,
        outputs=outputs,
    ) as run_id:
        result = await retriever.retrieve(request)
        outputs.update({
            "num_results": result.num_results,
            "cached": result.cached,
            "retrieval_mode": result.retrieval_mode,
            "processing_time_ms": result.processing_time_ms,
            "scores": [r.score for r in result.results],
        })

    if run_id:
        result = result.model_copy(update={"trace_id": run_id})
    return result
