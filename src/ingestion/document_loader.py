"""Document loading utilities for PDF and plain text files."""

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class DocumentPage:
    text: str
    page_number: int
    source: str


def load_pdf(file_path: Path) -> list[DocumentPage]:
    """Extract text from a PDF file, page by page."""
    reader = PdfReader(file_path)
    pages: list[DocumentPage] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(
                DocumentPage(
                    text=text,
                    page_number=i + 1,
                    source=file_path.name,
                )
            )

    if not pages:
        raise ValueError(f"No text content extracted from PDF: {file_path.name}")

    logger.info(f"Loaded {len(pages)} pages from PDF: {file_path.name}")
    return pages


def load_text(file_path: Path) -> list[DocumentPage]:
    """Read a plain text file as a single page."""
    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Empty text file: {file_path.name}")

    logger.info(f"Loaded text file: {file_path.name} ({len(text)} chars)")
    return [DocumentPage(text=text, page_number=1, source=file_path.name)]


def load_document(file_path: Path, content_type: str) -> list[DocumentPage]:
    """Route to the appropriate loader based on content type."""
    loaders = {
        "application/pdf": load_pdf,
        "text/plain": load_text,
    }

    loader = loaders.get(content_type)
    if loader is None:
        raise ValueError(f"Unsupported content type: {content_type}")

    return loader(file_path)
