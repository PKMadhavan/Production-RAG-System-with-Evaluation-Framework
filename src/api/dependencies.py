"""FastAPI dependency injection functions."""

from typing import AsyncGenerator, Optional

from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        yield session


def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding_service


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_redis(request: Request) -> Optional[Redis]:
    return request.app.state.redis_client
