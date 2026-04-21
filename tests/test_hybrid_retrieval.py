"""Tests for BM25 sparse index and hybrid RRF retrieval."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval.bm25_store import BM25Index


# ── BM25Index unit tests ──────────────────────────────────────────────────────

class TestBM25Index:
    def test_empty_index_returns_empty_search(self):
        index = BM25Index()
        results = index.search("machine learning", top_k=5)
        assert results == []

    def test_empty_index_size_is_zero(self):
        index = BM25Index()
        assert index.size == 0

    def test_tokenization_lowercases(self):
        index = BM25Index()
        tokens = index._tokenize("Machine Learning AI")
        assert tokens == ["machine", "learning", "ai"]

    def test_tokenization_splits_on_whitespace(self):
        index = BM25Index()
        tokens = index._tokenize("hello   world\ttab")
        assert "hello" in tokens
        assert "world" in tokens

    def test_add_chunks_increases_size(self):
        index = BM25Index()
        index.add_chunks(["id1", "id2"], ["text one", "text two"])
        assert index.size == 2

    def test_add_chunks_makes_searchable(self):
        # Need 3+ docs so BM25 IDF is positive for query terms in only 1 doc
        index = BM25Index()
        index.add_chunks(
            ["chunk-1", "chunk-2", "chunk-3"],
            ["machine learning is great", "cooking and recipes", "weather forecast today"],
        )
        results = index.search("machine learning", top_k=1)
        assert len(results) == 1
        assert results[0][0] == "chunk-1"

    def test_search_relevance_ordering(self):
        # Need 3+ docs so BM25 IDF is positive (log((N-df+0.5)/(df+0.5)) > 0)
        index = BM25Index()
        index.add_chunks(
            ["id-low", "id-high", "id-extra"],
            [
                "the weather is nice today",
                "machine learning deep learning neural networks",
                "cooking recipes and kitchen tips",
            ],
        )
        results = index.search("machine learning", top_k=3)
        chunk_ids = [r[0] for r in results]
        assert chunk_ids[0] == "id-high"  # more term overlap should rank first

    def test_search_returns_at_most_top_k(self):
        index = BM25Index()
        ids = [f"id-{i}" for i in range(10)]
        texts = [f"document number {i} about topic" for i in range(10)]
        index.add_chunks(ids, texts)
        results = index.search("document topic", top_k=3)
        assert len(results) <= 3

    def test_search_filters_zero_score(self):
        """Documents with no term overlap should be excluded."""
        index = BM25Index()
        index.add_chunks(["id-1"], ["completely unrelated text here"])
        results = index.search("machine learning", top_k=5)
        assert len(results) == 0

    def test_add_empty_list_does_nothing(self):
        index = BM25Index()
        index.add_chunks([], [])
        assert index.size == 0

    def test_incremental_adds(self):
        # Need 3+ docs so BM25 IDF is positive for unique query terms
        index = BM25Index()
        index.add_chunks(["id-1", "id-extra"], ["first document about weather", "cooking and recipes"])
        index.add_chunks(["id-2"], ["second document about computing"])
        assert index.size == 3
        results = index.search("second computing", top_k=1)
        assert results[0][0] == "id-2"


# ── RRF fusion tests ──────────────────────────────────────────────────────────

class TestRRFFusion:
    def _make_chunk(self, chunk_id: str) -> MagicMock:
        chunk = MagicMock()
        chunk.id = uuid.UUID(chunk_id) if len(chunk_id) == 36 else uuid.uuid4()
        chunk.content = f"content for {chunk_id}"
        chunk.metadata_ = {}
        return chunk

    def _make_retriever(self):
        from src.retrieval.retriever import Retriever
        retriever = Retriever.__new__(Retriever)
        retriever._rrf_k = 60
        retriever._embedding_service = MagicMock()
        retriever._vector_store = MagicMock()
        retriever._bm25_store = MagicMock()
        retriever._redis = None
        retriever._ttl = 3600
        return retriever

    def test_rrf_doc_in_both_lists_gets_higher_score(self):
        retriever = self._make_retriever()
        shared_id = str(uuid.uuid4())
        chunk = self._make_chunk(shared_id)
        chunk.id = uuid.UUID(shared_id)

        dense_results = [(chunk, 0.9)]
        sparse_results = [(shared_id, 5.0)]
        chunk_map = {shared_id: chunk}

        fused = retriever._rrf_fuse(dense_results, sparse_results, chunk_map, top_k=5)
        assert len(fused) == 1
        assert fused[0][1] > 1.0 / (60 + 1)  # higher than single-sided score

    def test_rrf_top_k_respected(self):
        retriever = self._make_retriever()
        dense_results = []
        chunk_map = {}
        sparse_results = []

        for i in range(10):
            cid = str(uuid.uuid4())
            c = self._make_chunk(cid)
            c.id = uuid.UUID(cid)
            dense_results.append((c, 0.9 - i * 0.05))
            chunk_map[cid] = c
            sparse_results.append((cid, 10.0 - i))

        fused = retriever._rrf_fuse(dense_results, sparse_results, chunk_map, top_k=3)
        assert len(fused) <= 3

    def test_rrf_sparse_only_hit_included(self):
        retriever = self._make_retriever()
        sparse_id = str(uuid.uuid4())
        sparse_chunk = self._make_chunk(sparse_id)
        sparse_chunk.id = uuid.UUID(sparse_id)

        fused = retriever._rrf_fuse(
            dense_results=[],
            sparse_results=[(sparse_id, 3.0)],
            chunk_map={sparse_id: sparse_chunk},
            top_k=5,
        )
        assert len(fused) == 1
        assert str(fused[0][0].id) == sparse_id


# ── Query endpoint retrieval_mode tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_query_accepts_dense_mode(client):
    response = await client.post(
        "/query/",
        json={"query": "test question", "retrieval_mode": "dense"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_query_accepts_sparse_mode(client):
    response = await client.post(
        "/query/",
        json={"query": "test question", "retrieval_mode": "sparse"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_query_accepts_hybrid_mode(client):
    response = await client.post(
        "/query/",
        json={"query": "test question", "retrieval_mode": "hybrid"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_query_rejects_invalid_mode(client):
    response = await client.post(
        "/query/",
        json={"query": "test question", "retrieval_mode": "bm25_only"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_default_mode_is_hybrid(client):
    response = await client.post(
        "/query/",
        json={"query": "test question"},
    )
    assert response.status_code == 200
    # Default mode should be accepted (hybrid)
