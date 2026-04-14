"""High-level retrieval orchestrator with caching."""

import hashlib
import json
import logging
import time
from typing import Optional

from redis.asyncio import Redis

from src.config import Settings
from src.models.schemas import QueryRequest, QueryResponse, RetrievedChunk
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Orchestrates query embedding, vector search, and result caching."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        redis_client: Optional[Redis] = None,
        settings: Optional[Settings] = None,
    ):
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._redis = redis_client
        self._ttl = settings.redis_ttl_query if settings else 3600

    def _cache_key(self, request: QueryRequest) -> str:
        key_str = f"{request.query}:{request.top_k}:{request.score_threshold}"
        return f"rag:query:{hashlib.sha256(key_str.encode()).hexdigest()}"

    async def retrieve(self, request: QueryRequest) -> QueryResponse:
        """Execute the full retrieval pipeline.

        1. Check Redis cache
        2. Embed query
        3. Vector similarity search
        4. Format and cache results
        """
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
                    logger.info(f"Cache hit for query: {request.query[:50]}...")
                    return response
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        # Embed query
        query_embedding = await self._embedding_service.embed_query(request.query)

        # Search
        results = await self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )

        # Format response
        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=str(chunk.id),
                content=chunk.content,
                score=round(score, 4),
                metadata=chunk.metadata_,
            )
            for chunk, score in results
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000
        response = QueryResponse(
            query=request.query,
            results=retrieved_chunks,
            num_results=len(retrieved_chunks),
            cached=False,
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
            f"Query completed: {len(retrieved_chunks)} results in {elapsed_ms:.1f}ms"
        )
        return response
