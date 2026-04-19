"""LangSmith tracing service for RAG pipeline observability."""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


class TracingService:
    """Wraps LangSmith client with async-safe, non-raising trace spans.

    All LangSmith API calls run via asyncio.to_thread() to avoid blocking
    the event loop. All failures are caught and logged — tracing errors
    never surface to the caller.

    Usage:
        outputs: dict = {}
        async with tracing.trace_query("what is ML?", "hybrid", 5, outputs) as run_id:
            result = await retriever.retrieve(request)
            outputs["num_results"] = result.num_results
        # run_id is a UUID string, or None if tracing failed
    """

    def __init__(self, client, project_name: str):
        self._client = client
        self._project = project_name

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _create_run(self, run_id: str, name: str, inputs: dict) -> bool:
        """Create a LangSmith run. Returns False if creation failed."""
        try:
            await asyncio.to_thread(
                self._client.create_run,
                id=run_id,
                name=name,
                run_type="chain",
                project_name=self._project,
                inputs=inputs,
                start_time=time.time(),
            )
            return True
        except Exception as e:
            logger.warning(f"LangSmith create_run failed: {e}")
            return False

    async def _update_run(
        self,
        run_id: str,
        outputs: dict,
        error: Optional[str] = None,
    ) -> None:
        """Update a LangSmith run with outputs. Never raises."""
        try:
            await asyncio.to_thread(
                self._client.update_run,
                run_id,
                outputs=outputs,
                error=error,
                end_time=time.time(),
            )
        except Exception as e:
            logger.warning(f"LangSmith update_run failed: {e}")

    # ── Trace spans ──────────────────────────────────────────────────────────

    @asynccontextmanager
    async def trace_query(
        self,
        query: str,
        retrieval_mode: str,
        top_k: int,
        outputs: dict,
    ) -> AsyncIterator[Optional[str]]:
        """Trace a /query request. Yields run_id or None on failure."""
        run_id = str(uuid.uuid4())
        created = await self._create_run(
            run_id=run_id,
            name="rag-query",
            inputs={
                "query": query,
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
            },
        )
        if not created:
            yield None
            return

        error_str: Optional[str] = None
        try:
            yield run_id
        except Exception as exc:
            error_str = str(exc)
            raise
        finally:
            await self._update_run(run_id, outputs, error_str)

    @asynccontextmanager
    async def trace_ingest(
        self,
        filename: str,
        chunking_strategy: str,
        chunk_size: int,
        outputs: dict,
    ) -> AsyncIterator[Optional[str]]:
        """Trace a /ingest request. Yields run_id or None on failure."""
        run_id = str(uuid.uuid4())
        created = await self._create_run(
            run_id=run_id,
            name="rag-ingest",
            inputs={
                "filename": filename,
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
            },
        )
        if not created:
            yield None
            return

        error_str: Optional[str] = None
        try:
            yield run_id
        except Exception as exc:
            error_str = str(exc)
            raise
        finally:
            await self._update_run(run_id, outputs, error_str)

    @asynccontextmanager
    async def trace_evaluate(
        self,
        num_samples: int,
        retrieval_mode: str,
        outputs: dict,
    ) -> AsyncIterator[Optional[str]]:
        """Trace a /evaluate request. Yields run_id or None on failure."""
        run_id = str(uuid.uuid4())
        created = await self._create_run(
            run_id=run_id,
            name="rag-evaluate",
            inputs={
                "num_samples": num_samples,
                "retrieval_mode": retrieval_mode,
            },
        )
        if not created:
            yield None
            return

        error_str: Optional[str] = None
        try:
            yield run_id
        except Exception as exc:
            error_str = str(exc)
            raise
        finally:
            await self._update_run(run_id, outputs, error_str)

    # ── Health probe ─────────────────────────────────────────────────────────

    async def check_connection(self) -> bool:
        """Verify LangSmith connectivity. Used by /health endpoint."""
        try:
            await asyncio.to_thread(
                lambda: list(self._client.list_projects(limit=1))
            )
            return True
        except Exception as e:
            logger.warning(f"LangSmith health check failed: {e}")
            return False
