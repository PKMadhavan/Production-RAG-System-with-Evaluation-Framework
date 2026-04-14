"""Unit tests for chunking strategies."""

import pytest

from src.ingestion.chunking import fixed_size_chunk
from src.ingestion.document_loader import DocumentPage


class TestFixedSizeChunk:
    def test_basic_chunking(self):
        """Text should be split into multiple chunks."""
        pages = [
            DocumentPage(
                text="word " * 200,  # ~1000 chars
                page_number=1,
                source="test.txt",
            )
        ]
        chunks = fixed_size_chunk(pages, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_preserves_source_metadata(self):
        """Each chunk should carry the source filename."""
        pages = [
            DocumentPage(text="Hello world", page_number=1, source="doc.pdf")
        ]
        chunks = fixed_size_chunk(pages, chunk_size=5000, chunk_overlap=0)
        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "doc.pdf"

    def test_preserves_page_number(self):
        """Each chunk should carry the page number from its source page."""
        pages = [
            DocumentPage(text="Page three content", page_number=3, source="doc.pdf")
        ]
        chunks = fixed_size_chunk(pages, chunk_size=5000, chunk_overlap=0)
        assert chunks[0].metadata["page_number"] == 3

    def test_chunk_indices_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        pages = [
            DocumentPage(text="word " * 200, page_number=1, source="test.txt")
        ]
        chunks = fixed_size_chunk(pages, chunk_size=100, chunk_overlap=10)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_multi_page_chunking(self):
        """Chunks from multiple pages should all be indexed sequentially."""
        pages = [
            DocumentPage(text="First page. " * 50, page_number=1, source="test.pdf"),
            DocumentPage(text="Second page. " * 50, page_number=2, source="test.pdf"),
        ]
        chunks = fixed_size_chunk(pages, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 2
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_small_text_single_chunk(self):
        """Text smaller than chunk_size should produce exactly one chunk."""
        pages = [
            DocumentPage(text="Short text.", page_number=1, source="test.txt")
        ]
        chunks = fixed_size_chunk(pages, chunk_size=5000, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_metadata_contains_strategy(self):
        """Metadata should include the chunking strategy name."""
        pages = [
            DocumentPage(text="Some text here.", page_number=1, source="test.txt")
        ]
        chunks = fixed_size_chunk(pages, chunk_size=5000, chunk_overlap=0)
        assert chunks[0].metadata["chunking_strategy"] == "fixed"

    def test_metadata_contains_chunk_params(self):
        """Metadata should record the chunk_size and chunk_overlap used."""
        pages = [
            DocumentPage(text="Some text here.", page_number=1, source="test.txt")
        ]
        chunks = fixed_size_chunk(pages, chunk_size=300, chunk_overlap=30)
        assert chunks[0].metadata["chunk_size"] == 300
        assert chunks[0].metadata["chunk_overlap"] == 30

    def test_empty_pages_list(self):
        """An empty pages list should return an empty chunks list."""
        chunks = fixed_size_chunk([], chunk_size=512, chunk_overlap=50)
        assert chunks == []
