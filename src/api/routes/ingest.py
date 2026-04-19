"""Document ingestion endpoint."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal, Optional


@asynccontextmanager
async def _noop_context() -> AsyncIterator[None]:
    """No-op async context manager used when tracing is disabled."""
    yield None

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_bm25_store,
    get_db_session,
    get_embedding_service,
    get_redis,
    get_settings,
    get_tracing_service,
)
from src.config import Settings
from src.ingestion.pipeline import ingest_document
from src.models.schemas import IngestResponse
from src.observability.tracing import TracingService
from src.retrieval.bm25_store import BM25Index
from src.retrieval.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"application/pdf", "text/plain"}


@router.post("/", response_model=IngestResponse)
async def ingest_document_endpoint(
    file: UploadFile = File(...),
    chunking_strategy: Literal["fixed", "semantic"] = Form("fixed"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    settings: Settings = Depends(get_settings),
    redis: Optional[Redis] = Depends(get_redis),
    bm25_store: BM25Index = Depends(get_bm25_store),
    tracing: Optional[TracingService] = Depends(get_tracing_service),
) -> IngestResponse:
    """Ingest a document (PDF or plain text) into the vector store."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
            f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    if chunk_size < 100 or chunk_size > 2000:
        raise HTTPException(
            status_code=400,
            detail="chunk_size must be between 100 and 2000",
        )
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be >= 0 and less than chunk_size",
        )

    try:
        outputs: dict = {}
        async with (
            tracing.trace_ingest(
                filename=file.filename or "unknown",
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                outputs=outputs,
            ) if tracing else _noop_context()
        ) as run_id:
            result = await ingest_document(
                file=file,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_service=embedding_service,
                session=session,
                settings=settings,
            )

            # Update BM25 index with new chunks
            chunk_ids = [c.chunk_id for c in result.chunks]
            chunk_texts = [c.content_preview for c in result.chunks]
            bm25_store.add_chunks(chunk_ids, chunk_texts)

            # Invalidate query cache since corpus changed
            if redis:
                try:
                    keys = [
                        key async for key in redis.scan_iter(match="rag:query:*")
                    ]
                    if keys:
                        await redis.delete(*keys)
                        logger.info(f"Invalidated {len(keys)} cached queries")
                except Exception as e:
                    logger.warning(f"Failed to invalidate query cache: {e}")

            outputs.update({
                "document_id": result.document_id,
                "num_chunks": result.num_chunks,
                "processing_time_ms": result.processing_time_ms,
            })

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
