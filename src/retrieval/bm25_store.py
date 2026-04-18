"""In-memory BM25 sparse index over document chunk corpus."""

import logging
from typing import Optional

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.database import DocumentChunk

logger = logging.getLogger(__name__)


class BM25Index:
    """Maintains an in-memory BM25 index mirroring the pgvector corpus.

    Lifecycle:
    - Built from DB on startup via build_from_db()
    - Extended incrementally after each successful ingest via add_chunks()
    - Queried synchronously via search() (caller wraps in asyncio.to_thread)
    """

    def __init__(self) -> None:
        self._corpus_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    async def build_from_db(
        cls, session_factory: async_sessionmaker[AsyncSession]
    ) -> "BM25Index":
        """Load all chunks from DB and build the BM25 index.

        Called once during lifespan startup. O(N * avg_doc_len).
        """
        index = cls()
        async with session_factory() as session:
            result = await session.execute(
                select(DocumentChunk.id, DocumentChunk.content)
            )
            rows = result.all()

        if rows:
            index._corpus_ids = [str(row[0]) for row in rows]
            index._tokenized_corpus = [index._tokenize(row[1]) for row in rows]
            index._rebuild()
            logger.info(f"BM25 index built: {len(rows)} chunks loaded from DB")
        else:
            logger.info("BM25 index empty: no chunks in DB yet")

        return index

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase whitespace tokenization. No external NLTK dependency."""
        return text.lower().split()

    def _rebuild(self) -> None:
        """Reconstruct BM25Okapi from the current corpus.

        The reference swap is atomic under CPython's GIL,
        so concurrent reads always see a valid (old or new) index.
        """
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Extend the index with newly ingested chunks and rebuild.

        Called after every successful ingest. O(N_total) rebuild is
        acceptable for Phase 2 corpus sizes (< 500k chunks).
        """
        if not chunk_ids:
            return

        self._corpus_ids.extend(chunk_ids)
        self._tokenized_corpus.extend(self._tokenize(t) for t in texts)
        self._rebuild()
        logger.info(
            f"BM25 index updated: {len(self._corpus_ids)} total chunks "
            f"(+{len(chunk_ids)} new)"
        )

    # ── Query ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25_score) tuples for the top_k matches.

        Runs synchronously — callers use asyncio.to_thread() to avoid
        blocking the event loop. Returns empty list if index is empty.
        """
        if self._bm25 is None or not self._corpus_ids:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get indices sorted by descending score
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            (self._corpus_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0  # skip zero-score (no term overlap)
        ]

    @property
    def size(self) -> int:
        """Number of chunks currently indexed."""
        return len(self._corpus_ids)
