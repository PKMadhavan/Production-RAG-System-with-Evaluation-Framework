"""Query endpoint for RAG retrieval."""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_retriever
from src.models.schemas import QueryRequest, QueryResponse
from src.retrieval.retriever import Retriever

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
) -> QueryResponse:
    """Query the RAG system and retrieve relevant document chunks."""
    return await retriever.retrieve(request)
