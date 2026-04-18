"""High-level retrieval orchestrator supporting dense, sparse, and hybrid modes."""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Optional

from redis.asyncio import Redis
from sqlalchemy import select

from src.config import Settings
from src.models.database import DocumentChunk
from src.models.schemas import QueryRequest, QueryResponse, RetrievedChunk
from src.retrieval.bm25_store import BM25Index
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Orchestrates dense, sparse, and hybrid retrieval with caching.

    Modes:
    - dense:  pgvector cosine similarity only
    - sparse: BM25 keyword matching only (no embedding call)
    - hybrid: RRF fusion of dense + sparse (default)
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        bm25_store: Optional[BM25Index] = None,
        redis_client: Optional[Redis] = None,
        settings: Optional[Settings] = None,
    ):
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._redis = redis_client
        self._ttl = settings.redis_ttl_query if settings else 3600
        self._rrf_k = settings.bm25_k_param if settings else 60

    # ── Cache ────────────────────────────────────────────────────────────────

    def _cache_key(self, request: QueryRequest) -> str:
        key_str = (
            f"{request.query}:{request.top_k}:"
            f"{request.score_threshold}:{request.retrieval_mode}"
        )
        return f"rag:query:{hashlib.sha256(key_str.encode()).hexdigest()}"

    # ── Sparse retrieval ─────────────────────────────────────────────────────

    async def _sparse_retrieve(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """Run BM25 search off the event loop. Returns (chunk_id, score) pairs."""
        if self._bm25_store is None:
            return []
        return await asyncio.to_thread(self._bm25_store.search, query, top_k)

    async def _fetch_chunks_by_ids(
        self, chunk_ids: list[str]
    ) -> dict[str, DocumentChunk]:
        """Bulk-fetch DocumentChunk objects by ID from the DB."""
        if not chunk_ids:
            return {}
        uuids = [uuid.UUID(cid) for cid in chunk_ids]
        async with self._vector_store._session_factory() as session:
            result = await session.execute(
                select(DocumentChunk).where(
                    DocumentChunk.id.in_(uuids)
                )
            )
            chunks = result.scalars().all()
        return {str(c.id): c for c in chunks}

    # ── RRF fusion ───────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense_results: list[tuple[DocumentChunk, float]],
        sparse_results: list[tuple[str, float]],
        chunk_map: dict[str, DocumentChunk],
        top_k: int,
    ) -> list[tuple[DocumentChunk, float]]:
        """Reciprocal Rank Fusion of dense and sparse result lists.

        rrf_score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_sparse(d))
        k=60 per Cormack et al. 2009.
        """
        k = self._rrf_k
        scores: dict[str, float] = {}
        chunks: dict[str, DocumentChunk] = {}

        # Dense contributions
        for rank, (chunk, _) in enumerate(dense_results):
            cid = str(chunk.id)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunks[cid] = chunk

        # Sparse contributions
        for rank, (cid, _) in enumerate(sparse_results):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunks and cid in chunk_map:
                chunks[cid] = chunk_map[cid]

        # Sort by RRF score descending, return top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (chunks[cid], score)
            for cid, score in ranked
            if cid in chunks
        ]

    # ── Main retrieve ─────────────────────────────────────────────────────────

    async def retrieve(self, request: QueryRequest) -> QueryResponse:
        """Execute the full retrieval pipeline for the requested mode."""
        start = time.perf_counter()

        # Check cache
        if self._redis:
            try:
                cached = await self._redis.get(self._cache_key(request))
                if cached:
                    response = QueryResponse(**json.loads(cached))
                    response.cached = True
                    response.processing_time_ms = (
                        (time.perf_counter() - start) * 1000
                    )
                    logger.info(f"Cache hit [{request.retrieval_mode}]: {request.query[:50]}")
                    return response
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        mode = request.retrieval_mode

        # ── Dense only ───────────────────────────────────────────────────────
        if mode == "dense":
            query_embedding = await self._embedding_service.embed_query(request.query)
            raw_results = await self._vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
            )
            retrieved = [
                RetrievedChunk(
                    chunk_id=str(c.id),
                    content=c.content,
                    score=round(s, 4),
                    metadata=c.metadata_,
                )
                for c, s in raw_results
            ]

        # ── Sparse only ──────────────────────────────────────────────────────
        elif mode == "sparse":
            sparse_hits = await self._sparse_retrieve(request.query, request.top_k)
            if sparse_hits:
                chunk_ids = [cid for cid, _ in sparse_hits]
                chunk_map = await self._fetch_chunks_by_ids(chunk_ids)
                retrieved = [
                    RetrievedChunk(
                        chunk_id=cid,
                        content=chunk_map[cid].content,
                        score=round(score, 4),
                        metadata=chunk_map[cid].metadata_,
                    )
                    for cid, score in sparse_hits
                    if cid in chunk_map
                ]
            else:
                retrieved = []

        # ── Hybrid (RRF) ─────────────────────────────────────────────────────
        else:
            query_embedding, sparse_hits = await asyncio.gather(
                self._embedding_service.embed_query(request.query),
                self._sparse_retrieve(request.query, request.top_k),
            )
            dense_results = await self._vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
            )

            # Fetch chunks that appear in sparse but not dense results
            dense_ids = {str(c.id) for c, _ in dense_results}
            sparse_only_ids = [
                cid for cid, _ in sparse_hits if cid not in dense_ids
            ]
            chunk_map = await self._fetch_chunks_by_ids(sparse_only_ids)

            fused = self._rrf_fuse(
                dense_results=dense_results,
                sparse_results=sparse_hits,
                chunk_map=chunk_map,
                top_k=request.top_k,
            )
            retrieved = [
                RetrievedChunk(
                    chunk_id=str(c.id),
                    content=c.content,
                    score=round(s, 4),
                    metadata=c.metadata_,
                )
                for c, s in fused
            ]

        elapsed_ms = (time.perf_counter() - start) * 1000
        response = QueryResponse(
            query=request.query,
            results=retrieved,
            num_results=len(retrieved),
            cached=False,
            retrieval_mode=mode,
            processing_time_ms=round(elapsed_ms, 2),
        )

        # Cache result
        if self._redis:
            try:
                await self._redis.setex(
                    self._cache_key(request),
                    self._ttl,
                    response.model_dump_json(),
                )
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")

        logger.info(
            f"[{mode}] {len(retrieved)} results in {elapsed_ms:.1f}ms — "
            f"'{request.query[:50]}'"
        )
        return response
