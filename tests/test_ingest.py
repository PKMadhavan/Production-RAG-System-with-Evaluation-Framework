"""Tests for the document ingestion endpoint."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from src.models.schemas import ChunkInfo, IngestResponse


@pytest.mark.asyncio
async def test_ingest_text_file(client):
    """Uploading a text file should return a valid IngestResponse."""
    mock_response = IngestResponse(
        document_id=str(uuid.uuid4()),
        filename="test.txt",
        num_chunks=2,
        chunking_strategy="fixed",
        chunks=[
            ChunkInfo(chunk_id=str(uuid.uuid4()), content_preview="chunk 1", chunk_index=0),
            ChunkInfo(chunk_id=str(uuid.uuid4()), content_preview="chunk 2", chunk_index=1),
        ],
        processing_time_ms=100.0,
    )

    with patch("src.api.routes.ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = mock_response

        response = await client.post(
            "/ingest/",
            files={"file": ("test.txt", b"This is a test document with enough text.", "text/plain")},
            data={"chunking_strategy": "fixed", "chunk_size": "512", "chunk_overlap": "50"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["num_chunks"] == 2
    assert data["chunking_strategy"] == "fixed"
    assert len(data["chunks"]) == 2


@pytest.mark.asyncio
async def test_ingest_pdf_file(client):
    """Uploading a PDF should be accepted (content type check)."""
    mock_response = IngestResponse(
        document_id=str(uuid.uuid4()),
        filename="test.pdf",
        num_chunks=1,
        chunking_strategy="fixed",
        chunks=[
            ChunkInfo(chunk_id=str(uuid.uuid4()), content_preview="pdf content", chunk_index=0),
        ],
        processing_time_ms=50.0,
    )

    with patch("src.api.routes.ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = mock_response

        response = await client.post(
            "/ingest/",
            files={"file": ("test.pdf", b"%PDF-1.4 fake pdf", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["filename"] == "test.pdf"


@pytest.mark.asyncio
async def test_ingest_rejects_unsupported_type(client):
    """Non-PDF/text files should be rejected with 400."""
    response = await client.post(
        "/ingest/",
        files={"file": ("image.jpg", b"fake image data", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_ingest_rejects_invalid_chunk_size(client):
    """chunk_size outside bounds should be rejected."""
    response = await client.post(
        "/ingest/",
        files={"file": ("test.txt", b"some text", "text/plain")},
        data={"chunk_size": "50"},  # below minimum of 100
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_ingest_rejects_overlap_greater_than_size(client):
    """chunk_overlap >= chunk_size should be rejected."""
    response = await client.post(
        "/ingest/",
        files={"file": ("test.txt", b"some text", "text/plain")},
        data={"chunk_size": "200", "chunk_overlap": "200"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_ingest_semantic_strategy(client):
    """Semantic chunking strategy should be accepted."""
    mock_response = IngestResponse(
        document_id=str(uuid.uuid4()),
        filename="test.txt",
        num_chunks=1,
        chunking_strategy="semantic",
        chunks=[
            ChunkInfo(chunk_id=str(uuid.uuid4()), content_preview="semantic chunk", chunk_index=0),
        ],
        processing_time_ms=200.0,
    )

    with patch("src.api.routes.ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = mock_response

        response = await client.post(
            "/ingest/",
            files={"file": ("test.txt", b"A long enough document for semantic chunking.", "text/plain")},
            data={"chunking_strategy": "semantic"},
        )

    assert response.status_code == 200
    assert response.json()["chunking_strategy"] == "semantic"
