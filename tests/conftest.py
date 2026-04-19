"""Shared test fixtures."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.config import Settings
from src.models.schemas import QueryResponse, RetrievedChunk


@pytest.fixture
def settings():
    return Settings(
        postgres_host="localhost",
        postgres_port=5432,
        postgres_user="testuser",
        postgres_password="testpass",
        postgres_db="testdb",
        redis_host="localhost",
        redis_port=6379,
        embedding_model="BAAI/bge-large-en-v1.5",
        embedding_dimension=1024,
    )


@pytest.fixture
def mock_session():
    """Create a mock async database session."""
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock())
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add_all = MagicMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def mock_session_factory(mock_session):
    """Create a mock session factory that yields the mock session."""
    factory = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    factory.return_value = ctx
    return factory


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = MagicMock()
    service.embed_query = AsyncMock(return_value=[0.1] * 1024)
    service.embed_texts = AsyncMock(return_value=[[0.1] * 1024])
    service.embed_texts_sync = MagicMock(return_value=[[0.1] * 1024])
    return service


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock()
    store.similarity_search = AsyncMock(return_value=[])
    store.insert_chunks = AsyncMock()
    return store


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        return_value=QueryResponse(
            query="test",
            results=[
                RetrievedChunk(
                    chunk_id=str(uuid.uuid4()),
                    content="This is a test chunk",
                    score=0.95,
                    metadata={"source": "test.txt", "page_number": 1},
                )
            ],
            num_results=1,
            cached=False,
            processing_time_ms=42.0,
        )
    )
    return retriever


@pytest_asyncio.fixture
async def client(
    settings,
    mock_session_factory,
    mock_session,
    mock_embedding_service,
    mock_vector_store,
    mock_retriever,
):
    """Create a test client with mocked dependencies."""
    # Patch the lifespan to avoid real DB/Redis/model connections
    from src.api.main import create_app

    app = create_app()

    # Override app.state with mocks
    app.state.settings = settings
    app.state.session_factory = mock_session_factory
    app.state.embedding_service = mock_embedding_service
    app.state.vector_store = mock_vector_store
    app.state.retriever = mock_retriever
    app.state.redis_client = None
    app.state.tracing_service = None
    app.state.bm25_store = MagicMock()
    app.state.bm25_store.size = 0
    app.state.evaluator = MagicMock()

    # Override the dependency to use our mock session directly
    from src.api.dependencies import get_db_session

    async def override_get_db_session():
        yield mock_session

    app.dependency_overrides[get_db_session] = override_get_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
