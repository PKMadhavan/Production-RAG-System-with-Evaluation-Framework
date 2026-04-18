"""RAGAS evaluation endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_evaluator
from src.evaluation.evaluator import RAGASEvaluator
from src.models.schemas import EvaluationRequest, EvaluationResponse

router = APIRouter()


@router.post("/", response_model=EvaluationResponse)
async def evaluate_endpoint(
    request: EvaluationRequest,
    evaluator: RAGASEvaluator = Depends(get_evaluator),
) -> EvaluationResponse:
    """Run RAGAS evaluation over the live retrieval pipeline.

    Accepts a list of questions (+ optional ground truth answers).
    Returns faithfulness, answer_relevancy, and optionally context_recall
    and context_precision scores per sample and aggregated.
    """
    try:
        return await evaluator.evaluate(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
