"""HuggingFace embedding service with Redis caching."""

import asyncio
import hashlib
import json
import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps HuggingFace embeddings with optional Redis caching."""

    def __init__(
        self,
        model_name: str,
        redis_client: Optional[Redis] = None,
        ttl: int = 86400,
    ):
        logger.info(f"Loading embedding model: {model_name}")
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._redis = redis_client
        self._ttl = ttl
        logger.info(f"Embedding model loaded: {model_name}")

    def _cache_key(self, text: str) -> str:
        return f"rag:embed:{hashlib.sha256(text.encode()).hexdigest()}"

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Uses Redis cache if available."""
        # Check cache
        if self._redis:
            try:
                cached = await self._redis.get(self._cache_key(text))
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        # Compute embedding (run in thread to avoid blocking event loop)
        embedding = await asyncio.to_thread(
            self._model.embed_query, text
        )

        # Cache result
        if self._redis:
            try:
                await self._redis.setex(
                    self._cache_key(text),
                    self._ttl,
                    json.dumps(embedding),
                )
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")

        return embedding

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts for ingestion. No caching (batch operation)."""
        return await asyncio.to_thread(
            self._model.embed_documents, texts
        )

    def embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding for use in chunking (SemanticChunker)."""
        return self._model.embed_documents(texts)
