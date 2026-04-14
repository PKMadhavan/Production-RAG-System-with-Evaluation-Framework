"""pgvector-backed vector store for similarity search."""

import logging
import uuid

from sqlalchemy import delete, desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.database import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and similarity search via pgvector."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """Find the most similar chunks to the query embedding.

        Uses pgvector's cosine distance operator (<=>).
        Returns (chunk, score) tuples sorted by descending similarity.
        Score = 1 - cosine_distance (higher is more similar).
        """
        async with self._session_factory() as session:
            # Cosine distance: 0 = identical, 2 = opposite
            # Score = 1 - distance: 1 = identical, -1 = opposite
            cosine_distance = DocumentChunk.embedding.cosine_distance(query_embedding)
            score_expr = (1 - cosine_distance).label("score")

            stmt = (
                select(DocumentChunk, score_expr)
                .where((1 - cosine_distance) >= score_threshold)
                .order_by(desc("score"))
                .limit(top_k)
            )

            result = await session.execute(stmt)
            rows = result.all()

            logger.info(
                f"Similarity search: found {len(rows)} results "
                f"(top_k={top_k}, threshold={score_threshold})"
            )
            return [(row[0], float(row[1])) for row in rows]

    async def insert_chunks(
        self,
        chunks: list[DocumentChunk],
        session: AsyncSession,
    ) -> None:
        """Bulk insert document chunks. Caller manages the transaction."""
        session.add_all(chunks)
        await session.flush()

    async def delete_document(
        self,
        document_id: uuid.UUID,
        session: AsyncSession,
    ) -> int:
        """Delete all chunks for a given document. Returns count deleted."""
        stmt = (
            delete(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .returning(DocumentChunk.id)
        )
        result = await session.execute(stmt)
        deleted = len(result.all())
        await session.commit()
        logger.info(f"Deleted {deleted} chunks for document {document_id}")
        return deleted
