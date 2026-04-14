"""Tests for the query endpoint."""

import uuid
from unittest.mock import AsyncMock

import pytest

from src.models.schemas import QueryResponse, RetrievedChunk


@pytest.mark.asyncio
async def test_query_returns_results(client):
    """A valid query should return results from the retriever."""
    response = await client.post(
        "/query/",
        json={"query": "What is machine learning?", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert "num_results" in data
    assert data["num_results"] == 1
    assert data["results"][0]["score"] == 0.95


@pytest.mark.asyncio
async def test_query_empty_results(client, mock_retriever):
    """When no chunks match, should return empty results."""
    mock_retriever.retrieve = AsyncMock(
        return_value=QueryResponse(
            query="obscure question",
            results=[],
            num_results=0,
            cached=False,
            processing_time_ms=10.0,
        )
    )

    response = await client.post(
        "/query/",
        json={"query": "obscure question", "top_k": 5},
    )
    assert response.status_code == 200
    assert response.json()["num_results"] == 0
    assert response.json()["results"] == []


@pytest.mark.asyncio
async def test_query_empty_string_rejected(client):
    """Empty query string should fail validation."""
    response = await client.post(
        "/query/",
        json={"query": "", "top_k": 5},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_missing_body(client):
    """Missing request body should fail validation."""
    response = await client.post("/query/")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_default_top_k(client):
    """top_k should default to 5 when not specified."""
    response = await client.post(
        "/query/",
        json={"query": "test question"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_query_custom_parameters(client):
    """Custom top_k and score_threshold should be accepted."""
    response = await client.post(
        "/query/",
        json={
            "query": "test question",
            "top_k": 10,
            "score_threshold": 0.5,
        },
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_query_invalid_top_k(client):
    """top_k outside bounds should fail validation."""
    response = await client.post(
        "/query/",
        json={"query": "test", "top_k": 0},
    )
    assert response.status_code == 422

    response = await client.post(
        "/query/",
        json={"query": "test", "top_k": 100},
    )
    assert response.status_code == 422
