"""RAGAS evaluation endpoint."""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_evaluator, get_tracing_service
from src.evaluation.evaluator import RAGASEvaluator
from src.models.schemas import EvaluationRequest, EvaluationResponse
from src.observability.tracing import TracingService

router = APIRouter()


@asynccontextmanager
async def _noop_context() -> AsyncIterator[None]:
    yield None


@router.post("/", response_model=EvaluationResponse)
async def evaluate_endpoint(
    request: EvaluationRequest,
    evaluator: RAGASEvaluator = Depends(get_evaluator),
    tracing: Optional[TracingService] = Depends(get_tracing_service),
) -> EvaluationResponse:
    """Run RAGAS evaluation over the live retrieval pipeline."""
    try:
        outputs: dict = {}
        async with (
            tracing.trace_evaluate(
                num_samples=len(request.samples),
                retrieval_mode=request.retrieval_mode,
                outputs=outputs,
            ) if tracing else _noop_context()
        ) as run_id:
            result = await evaluator.evaluate(request)
            outputs.update({
                "faithfulness": result.aggregate_scores.faithfulness,
                "answer_relevancy": result.aggregate_scores.answer_relevancy,
                "context_recall": result.aggregate_scores.context_recall,
                "context_precision": result.aggregate_scores.context_precision,
                "processing_time_ms": result.processing_time_ms,
                "llm_used": result.llm_used,
            })

        if run_id:
            result = result.model_copy(update={"trace_id": run_id})
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
