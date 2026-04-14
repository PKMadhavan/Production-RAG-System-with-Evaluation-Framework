"""Document chunking strategies: fixed-size and semantic."""

import logging
from dataclasses import dataclass, field
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.document_loader import DocumentPage

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def fixed_size_chunk(
    pages: list[DocumentPage],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[ChunkResult]:
    """Split documents into fixed-size chunks with overlap.

    Uses RecursiveCharacterTextSplitter with smart separators
    to avoid splitting mid-sentence when possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[ChunkResult] = []
    chunk_index = 0

    for page in pages:
        splits = splitter.split_text(page.text)
        for split_text in splits:
            chunks.append(
                ChunkResult(
                    text=split_text,
                    chunk_index=chunk_index,
                    metadata={
                        "source": page.source,
                        "page_number": page.page_number,
                        "chunking_strategy": "fixed",
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    },
                )
            )
            chunk_index += 1

    logger.info(
        f"Fixed-size chunking: {len(pages)} pages -> {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def semantic_chunk(
    pages: list[DocumentPage],
    embedding_function: Callable[[list[str]], list[list[float]]],
) -> list[ChunkResult]:
    """Split documents into semantically coherent chunks.

    Groups sentences by semantic similarity using embeddings.
    Falls back to fixed-size chunking if semantic chunking fails.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_core.embeddings import Embeddings

        class _EmbeddingWrapper(Embeddings):
            """Wraps a callable into a LangChain Embeddings interface."""

            def __init__(self, fn: Callable[[list[str]], list[list[float]]]):
                self._fn = fn

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self._fn(texts)

            def embed_query(self, text: str) -> list[float]:
                return self._fn([text])[0]

        wrapper = _EmbeddingWrapper(embedding_function)
        splitter = SemanticChunker(
            embeddings=wrapper,
            breakpoint_threshold_type="percentile",
        )

        chunks: list[ChunkResult] = []
        chunk_index = 0

        for page in pages:
            splits = splitter.split_text(page.text)
            for split_text in splits:
                chunks.append(
                    ChunkResult(
                        text=split_text,
                        chunk_index=chunk_index,
                        metadata={
                            "source": page.source,
                            "page_number": page.page_number,
                            "chunking_strategy": "semantic",
                        },
                    )
                )
                chunk_index += 1

        logger.info(
            f"Semantic chunking: {len(pages)} pages -> {len(chunks)} chunks"
        )
        return chunks

    except Exception as e:
        logger.warning(
            f"Semantic chunking failed ({e}), falling back to fixed-size"
        )
        return fixed_size_chunk(pages)
