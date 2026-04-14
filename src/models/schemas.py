"""Pydantic models for API request/response contracts."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    postgres_connected: bool = False
    redis_connected: bool = False


class ChunkInfo(BaseModel):
    chunk_id: str
    content_preview: str
    chunk_index: int


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    chunking_strategy: str
    chunks: list[ChunkInfo]
    processing_time_ms: float = 0.0


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    num_results: int
    cached: bool = False
    processing_time_ms: float = 0.0
