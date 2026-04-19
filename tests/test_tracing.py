"""Unit tests for TracingService."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.observability.tracing import TracingService


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.create_run = MagicMock()
    client.update_run = MagicMock()
    client.list_projects = MagicMock(return_value=iter([MagicMock()]))
    return client


@pytest.fixture
def tracing(mock_client):
    return TracingService(client=mock_client, project_name="test-project")


# ── trace_query ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trace_query_calls_create_run(tracing, mock_client):
    outputs = {}
    async with tracing.trace_query("what is ML?", "hybrid", 5, outputs) as run_id:
        outputs["num_results"] = 1

    mock_client.create_run.assert_called_once()
    call_kwargs = mock_client.create_run.call_args[1]
    assert call_kwargs["name"] == "rag-query"
    assert call_kwargs["inputs"]["query"] == "what is ML?"
    assert call_kwargs["inputs"]["retrieval_mode"] == "hybrid"


@pytest.mark.asyncio
async def test_trace_query_calls_update_run(tracing, mock_client):
    outputs = {}
    async with tracing.trace_query("test query", "dense", 3, outputs) as run_id:
        outputs["num_results"] = 2

    mock_client.update_run.assert_called_once()


@pytest.mark.asyncio
async def test_trace_query_yields_valid_uuid(tracing):
    outputs = {}
    async with tracing.trace_query("test", "hybrid", 5, outputs) as run_id:
        pass

    assert run_id is not None
    uuid.UUID(run_id)  # raises if not valid UUID


@pytest.mark.asyncio
async def test_trace_query_update_called_on_exception(tracing, mock_client):
    outputs = {}
    with pytest.raises(ValueError):
        async with tracing.trace_query("test", "hybrid", 5, outputs) as run_id:
            raise ValueError("something broke")

    mock_client.update_run.assert_called_once()
    call_kwargs = mock_client.update_run.call_args
    assert call_kwargs[1]["error"] == "something broke"


@pytest.mark.asyncio
async def test_trace_query_yields_none_when_create_fails(mock_client):
    mock_client.create_run.side_effect = Exception("network error")
    tracing = TracingService(client=mock_client, project_name="test")

    outputs = {}
    async with tracing.trace_query("test", "hybrid", 5, outputs) as run_id:
        pass

    assert run_id is None
    mock_client.update_run.assert_not_called()


# ── trace_ingest ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trace_ingest_calls_create_run(tracing, mock_client):
    outputs = {}
    async with tracing.trace_ingest("doc.txt", "fixed", 512, outputs) as run_id:
        outputs["num_chunks"] = 3

    mock_client.create_run.assert_called_once()
    call_kwargs = mock_client.create_run.call_args[1]
    assert call_kwargs["name"] == "rag-ingest"
    assert call_kwargs["inputs"]["filename"] == "doc.txt"


@pytest.mark.asyncio
async def test_trace_ingest_yields_valid_uuid(tracing):
    outputs = {}
    async with tracing.trace_ingest("doc.pdf", "semantic", 512, outputs) as run_id:
        pass
    assert run_id is not None
    uuid.UUID(run_id)


@pytest.mark.asyncio
async def test_trace_ingest_yields_none_when_create_fails(mock_client):
    mock_client.create_run.side_effect = Exception("auth error")
    tracing = TracingService(client=mock_client, project_name="test")

    outputs = {}
    async with tracing.trace_ingest("file.txt", "fixed", 512, outputs) as run_id:
        pass
    assert run_id is None


# ── trace_evaluate ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trace_evaluate_calls_create_run(tracing, mock_client):
    outputs = {}
    async with tracing.trace_evaluate(3, "hybrid", outputs) as run_id:
        outputs["faithfulness"] = 0.85

    mock_client.create_run.assert_called_once()
    call_kwargs = mock_client.create_run.call_args[1]
    assert call_kwargs["name"] == "rag-evaluate"
    assert call_kwargs["inputs"]["num_samples"] == 3


@pytest.mark.asyncio
async def test_trace_evaluate_yields_valid_uuid(tracing):
    outputs = {}
    async with tracing.trace_evaluate(2, "dense", outputs) as run_id:
        pass
    assert run_id is not None
    uuid.UUID(run_id)


# ── check_connection ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_connection_returns_true(tracing):
    result = await tracing.check_connection()
    assert result is True


@pytest.mark.asyncio
async def test_check_connection_returns_false_on_error(mock_client):
    mock_client.list_projects.side_effect = Exception("invalid API key")
    tracing = TracingService(client=mock_client, project_name="test")
    result = await tracing.check_connection()
    assert result is False


# ── Health endpoint with LangSmith ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_has_langsmith_connected_field(client):
    response = await client.get("/health")
    assert "langsmith_connected" in response.json()


@pytest.mark.asyncio
async def test_health_langsmith_false_when_no_service(client):
    """tracing_service is None in test fixtures."""
    response = await client.get("/health")
    assert response.json()["langsmith_connected"] is False


# ── Query endpoint with tracing ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_trace_id_none_when_no_tracing(client):
    """When tracing_service is None, trace_id should be None."""
    response = await client.post(
        "/query/",
        json={"query": "test question", "retrieval_mode": "hybrid"},
    )
    assert response.status_code == 200
    assert response.json()["trace_id"] is None
