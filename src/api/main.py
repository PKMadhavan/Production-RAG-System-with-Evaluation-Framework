"""FastAPI application factory with lifespan management."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from src.api.routes import health, ingest, query
from src.config import Settings
from src.models.database import create_engine, create_session_factory, init_db
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    settings = Settings()
    app.state.settings = settings

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Database
    logger.info("Initializing database connection...")
    engine = create_engine(settings)
    await init_db(engine)
    app.state.session_factory = create_session_factory(engine)
    logger.info("Database initialized")

    # Redis (optional — graceful degradation)
    redis_client = None
    try:
        redis_client = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
        await redis_client.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}), caching disabled")
        redis_client = None
    app.state.redis_client = redis_client

    # Embedding service
    logger.info("Loading embedding model (this may take a moment)...")
    app.state.embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
        redis_client=redis_client,
        ttl=settings.redis_ttl_embedding,
    )

    # Vector store & retriever
    app.state.vector_store = VectorStore(app.state.session_factory)
    app.state.retriever = Retriever(
        embedding_service=app.state.embedding_service,
        vector_store=app.state.vector_store,
        redis_client=redis_client,
        settings=settings,
    )

    logger.info("RAG API ready")
    yield

    # Shutdown
    logger.info("Shutting down...")
    await engine.dispose()
    if redis_client:
        await redis_client.close()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Production RAG API",
        description="Retrieval-Augmented Generation system with pgvector and HuggingFace embeddings",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health.router, tags=["health"])
    app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
    app.include_router(query.router, prefix="/query", tags=["query"])

    return app


app = create_app()
