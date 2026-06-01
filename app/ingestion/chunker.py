"""Chunk page streams into overlapping passages while tracking page ranges.

Character-based windows with overlap keep related sentences together and preserve
a page_start/page_end span so every chunk can cite the exact source pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .pdf_loader import Page


@dataclass
class ChunkSpec:
    content: str
    page_start: int
    page_end: int


def chunk_pages(
    pages: Iterable[Page], *, chunk_chars: int = 1200, overlap: int = 200
) -> List[ChunkSpec]:
    """Concatenate page text (with page markers) and slice into overlapping chunks."""
    if overlap >= chunk_chars:
        raise ValueError("overlap must be smaller than chunk_chars")

    # Build a single stream while recording the page each character belongs to.
    buf: List[str] = []
    char_page: List[int] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        if buf:
            buf.append("\n\n")
            char_page.extend([page.number, page.number])
        buf.append(text)
        char_page.extend([page.number] * len(text))

    stream = "".join(buf)
    n = len(stream)
    if n == 0:
        return []

    chunks: List[ChunkSpec] = []
    step = chunk_chars - overlap
    start = 0
    while start < n:
        end = min(start + chunk_chars, n)
        # Prefer to break on a whitespace boundary near the end for cleaner chunks.
        if end < n:
            window = stream.rfind(" ", start + step, end)
            if window != -1:
                end = window
        segment = stream[start:end].strip()
        if segment:
            ps = char_page[start] if start < len(char_page) else char_page[-1]
            pe = char_page[min(end, len(char_page)) - 1]
            chunks.append(ChunkSpec(content=segment, page_start=ps, page_end=pe))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
