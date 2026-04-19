"""Pydantic models for API request/response contracts."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    postgres_connected: bool = False
    redis_connected: bool = False
    langsmith_connected: bool = False


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
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = Field(default="hybrid")


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
    retrieval_mode: str = "hybrid"
    processing_time_ms: float = 0.0
    trace_id: Optional[str] = None


# ── Evaluation ──────────────────────────────────────────────────────────────

class EvaluationSample(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    ground_truth: Optional[str] = None


class EvaluationRequest(BaseModel):
    samples: list[EvaluationSample] = Field(..., min_length=1, max_length=100)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = Field(default="hybrid")


class MetricScores(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None


class EvaluationSampleResult(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None
    scores: MetricScores


class EvaluationResponse(BaseModel):
    num_samples: int
    aggregate_scores: MetricScores
    sample_results: list[EvaluationSampleResult]
    llm_used: str
    processing_time_ms: float = 0.0
    trace_id: Optional[str] = None
