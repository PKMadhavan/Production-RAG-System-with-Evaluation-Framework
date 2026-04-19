"""FastAPI application factory with lifespan management."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from src.api.routes import evaluate, health, ingest, query
from src.config import Settings
from src.evaluation.evaluator import RAGASEvaluator
from src.models.database import create_engine, create_session_factory, init_db
from src.observability.tracing import TracingService
from src.retrieval.bm25_store import BM25Index
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

    # Vector store
    app.state.vector_store = VectorStore(app.state.session_factory)

    # BM25 index — built from existing DB chunks on startup
    logger.info("Building BM25 index from existing document chunks...")
    bm25_store = await BM25Index.build_from_db(app.state.session_factory)
    app.state.bm25_store = bm25_store
    logger.info(f"BM25 index ready: {bm25_store.size} chunks indexed")

    # Retriever (now includes BM25)
    app.state.retriever = Retriever(
        embedding_service=app.state.embedding_service,
        vector_store=app.state.vector_store,
        bm25_store=bm25_store,
        redis_client=redis_client,
        settings=settings,
    )

    # RAGAS Evaluator
    app.state.evaluator = RAGASEvaluator(
        retriever=app.state.retriever,
        settings=settings,
    )
    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY not set: evaluation will use extractive fallback. "
            "Set OPENAI_API_KEY in .env for LLM-generated answers."
        )

    # LangSmith tracing (optional — graceful degradation)
    tracing_service = None
    if settings.langsmith_api_key:
        try:
            import langsmith
            client = langsmith.Client(api_key=settings.langsmith_api_key)
            tracing_service = TracingService(
                client=client,
                project_name=settings.langsmith_project,
            )
            logger.info(f"LangSmith tracing enabled (project: {settings.langsmith_project})")
        except Exception as e:
            logger.warning(f"LangSmith initialization failed ({e}), tracing disabled")
    else:
        logger.info("LANGSMITH_API_KEY not set, tracing disabled")
    app.state.tracing_service = tracing_service

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
        description=(
            "Retrieval-Augmented Generation system with pgvector, "
            "HuggingFace embeddings, BM25 hybrid retrieval, and RAGAS evaluation"
        ),
        version="0.2.0",
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
    app.include_router(evaluate.router, prefix="/evaluate", tags=["evaluation"])

    return app


app = create_app()
