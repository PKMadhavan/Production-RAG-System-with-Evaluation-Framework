"""RAGAS-based evaluation pipeline for the RAG system."""

import asyncio
import logging
import time
from typing import Optional

from src.config import Settings
from src.models.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationSampleResult,
    MetricScores,
    QueryRequest,
)
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """Runs RAGAS evaluation over the live retrieval pipeline.

    Answer generation modes:
    - OpenAI gpt-4o-mini (when OPENAI_API_KEY is set)
    - Extractive fallback: returns first retrieved context (no LLM required)
    """

    def __init__(self, retriever: Retriever, settings: Settings):
        self._retriever = retriever
        self._settings = settings
        self._use_openai = bool(settings.openai_api_key)
        self._llm_label = (
            f"openai/{settings.openai_model}"
            if self._use_openai
            else "extractive"
        )

    # ── Answer generation ────────────────────────────────────────────────────

    async def _generate_answer(self, question: str, contexts: list[str]) -> str:
        """Generate an answer from question + retrieved context passages."""
        if not contexts:
            return "No relevant context found."

        if self._use_openai:
            return await self._openai_answer(question, contexts)

        # Extractive fallback: return the first context, truncated
        return contexts[0][:500]

    async def _openai_answer(self, question: str, contexts: list[str]) -> str:
        """Use OpenAI to generate a grounded answer."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            joined = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
            prompt = (
                f"Answer the following question using only the provided context. "
                f"If the answer is not in the context, say 'I don't know.'\n\n"
                f"Context:\n{joined}\n\n"
                f"Question: {question}\n\nAnswer:"
            )

            llm = ChatOpenAI(
                model=self._settings.openai_model,
                api_key=self._settings.openai_api_key,
                temperature=0,
            )
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()

        except Exception as e:
            logger.warning(f"OpenAI answer generation failed ({e}), using extractive fallback")
            return contexts[0][:500]

    # ── RAGAS metrics ────────────────────────────────────────────────────────

    def _run_ragas_metrics(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        ground_truths: list[Optional[str]],
    ) -> list[MetricScores]:
        """Run RAGAS evaluation synchronously (called via asyncio.to_thread).

        Selects metrics based on ground_truth availability:
        - Always: faithfulness, answer_relevancy
        - With ground_truth: context_recall, context_precision
        """
        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import answer_relevancy, faithfulness

            has_ground_truth = any(gt for gt in ground_truths)
            metrics = [faithfulness, answer_relevancy]

            data: dict = {
                "question": questions,
                "answer": answers,
                "contexts": contexts_list,
            }

            if has_ground_truth:
                from ragas.metrics import context_precision, context_recall
                metrics += [context_recall, context_precision]
                data["ground_truth"] = [gt or "" for gt in ground_truths]

            dataset = Dataset.from_dict(data)
            result = ragas_evaluate(dataset, metrics=metrics)
            result_df = result.to_pandas()

            sample_scores: list[MetricScores] = []
            for _, row in result_df.iterrows():
                sample_scores.append(
                    MetricScores(
                        faithfulness=_safe_float(row.get("faithfulness")),
                        answer_relevancy=_safe_float(row.get("answer_relevancy")),
                        context_recall=_safe_float(row.get("context_recall")),
                        context_precision=_safe_float(row.get("context_precision")),
                    )
                )
            return sample_scores

        except ImportError:
            logger.error("RAGAS or datasets not installed. Run: pip install ragas datasets")
            # Return null scores rather than crashing
            return [MetricScores() for _ in questions]
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return [MetricScores() for _ in questions]

    # ── Main evaluate ────────────────────────────────────────────────────────

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """Run the full evaluation pipeline.

        1. Retrieve contexts for each question
        2. Generate answers (OpenAI or extractive)
        3. Run RAGAS metrics in a background thread
        4. Return structured evaluation report
        """
        start = time.perf_counter()

        questions: list[str] = []
        answers: list[str] = []
        contexts_list: list[list[str]] = []
        ground_truths: list[Optional[str]] = []

        # Step 1 & 2: retrieve + generate answers for each sample
        for sample in request.samples:
            query_req = QueryRequest(
                query=sample.question,
                top_k=request.top_k,
                retrieval_mode=request.retrieval_mode,
            )
            query_resp = await self._retriever.retrieve(query_req)
            contexts = [chunk.content for chunk in query_resp.results]
            answer = await self._generate_answer(sample.question, contexts)

            questions.append(sample.question)
            answers.append(answer)
            contexts_list.append(contexts)
            ground_truths.append(sample.ground_truth)

        # Step 3: run RAGAS metrics off the event loop
        sample_scores = await asyncio.to_thread(
            self._run_ragas_metrics,
            questions,
            answers,
            contexts_list,
            ground_truths,
        )

        # Step 4: build per-sample results and aggregate
        sample_results = [
            EvaluationSampleResult(
                question=q,
                answer=a,
                contexts=ctx,
                ground_truth=gt,
                scores=scores,
            )
            for q, a, ctx, gt, scores in zip(
                questions, answers, contexts_list, ground_truths, sample_scores
            )
        ]

        aggregate = _aggregate_scores(sample_scores)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Evaluation complete: {len(request.samples)} samples, "
            f"faithfulness={aggregate.faithfulness}, "
            f"answer_relevancy={aggregate.answer_relevancy}, "
            f"{elapsed_ms:.0f}ms"
        )

        return EvaluationResponse(
            num_samples=len(request.samples),
            aggregate_scores=aggregate,
            sample_results=sample_results,
            llm_used=self._llm_label,
            processing_time_ms=round(elapsed_ms, 2),
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(value) -> Optional[float]:
    """Convert a value to float, returning None if conversion fails."""
    try:
        f = float(value)
        return round(f, 4) if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


def _aggregate_scores(scores: list[MetricScores]) -> MetricScores:
    """Compute mean for each metric across all samples (ignoring None values)."""
    def mean_of(field: str) -> Optional[float]:
        vals = [getattr(s, field) for s in scores if getattr(s, field) is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 4)

    return MetricScores(
        faithfulness=mean_of("faithfulness"),
        answer_relevancy=mean_of("answer_relevancy"),
        context_recall=mean_of("context_recall"),
        context_precision=mean_of("context_precision"),
    )
