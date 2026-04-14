"""Document ingestion pipeline: load → chunk → embed → store."""

import logging
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.ingestion.chunking import ChunkResult, fixed_size_chunk, semantic_chunk
from src.ingestion.document_loader import load_document
from src.models.database import DocumentChunk
from src.models.schemas import ChunkInfo, IngestResponse
from src.retrieval.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


async def ingest_document(
    file: UploadFile,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_service: EmbeddingService,
    session: AsyncSession,
    settings: Settings,
) -> IngestResponse:
    """Orchestrate the full document ingestion pipeline.

    1. Save uploaded file to temp location
    2. Extract text (PDF or plain text)
    3. Chunk the text
    4. Embed all chunks in batches
    5. Store in pgvector
    6. Return response with chunk info
    """
    start = time.perf_counter()
    document_id = uuid.uuid4()
    tmp_path: Path | None = None

    try:
        # Save upload to temp file
        content = await file.read()
        if not content:
            raise ValueError("Uploaded file is empty")

        if len(content) > settings.max_file_size_bytes:
            raise ValueError(
                f"File exceeds maximum size of {settings.max_file_size_mb}MB"
            )

        suffix = Path(file.filename or "upload").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Load document
        pages = load_document(tmp_path, file.content_type or "text/plain")

        # Chunk
        if chunking_strategy == "semantic":
            chunks = semantic_chunk(
                pages,
                embedding_function=embedding_service.embed_texts_sync,
            )
        else:
            chunks = fixed_size_chunk(
                pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        if not chunks:
            raise ValueError("No chunks produced from document")

        # Embed in batches
        all_embeddings: list[list[float]] = []
        texts = [c.text for c in chunks]
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_embeddings = await embedding_service.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)

        # Create DB records
        db_chunks = [
            DocumentChunk(
                document_id=document_id,
                content=chunk.text,
                embedding=embedding,
                metadata_=chunk.metadata,
            )
            for chunk, embedding in zip(chunks, all_embeddings)
        ]

        session.add_all(db_chunks)
        await session.commit()

        # Build response
        chunk_infos = [
            ChunkInfo(
                chunk_id=str(db_chunk.id),
                content_preview=db_chunk.content[:200],
                chunk_index=chunk.chunk_index,
            )
            for db_chunk, chunk in zip(db_chunks, chunks)
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Ingested {file.filename}: {len(chunks)} chunks in {elapsed_ms:.1f}ms"
        )

        return IngestResponse(
            document_id=str(document_id),
            filename=file.filename or "unknown",
            num_chunks=len(chunks),
            chunking_strategy=chunking_strategy,
            chunks=chunk_infos,
            processing_time_ms=elapsed_ms,
        )

    except ValueError:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise RuntimeError(f"Document ingestion failed: {e}") from e
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
