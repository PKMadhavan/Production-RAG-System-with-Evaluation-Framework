"""Tests for the health check endpoint."""

import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_structure(client):
    response = await client.get("/health")
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "postgres_connected" in data
    assert "redis_connected" in data


@pytest.mark.asyncio
async def test_health_version(client):
    response = await client.get("/health")
    data = response.json()
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_health_redis_disconnected(client):
    """Redis is None in test fixtures, so redis_connected should be False."""
    response = await client.get("/health")
    data = response.json()
    assert data["redis_connected"] is False
