"""Extract per-page text from PDFs with light cleaning.

Uses PyMuPDF (``fitz``). All corpus PDFs were verified to be text-based, so no OCR
is required; if a page yields no text it is skipped (and surfaced in stats so a
scanned book can be flagged for OCR before ingestion).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Optional

_WS = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # de-hyphenate line breaks
    text = text.replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()


@dataclass
class Page:
    number: int      # 1-based page number
    text: str


def load_pdf_pages(
    path: str, *, max_pages: Optional[int] = None, start_page: int = 1
) -> Iterator[Page]:
    import fitz

    doc = fitz.open(path)
    try:
        total = doc.page_count
        last = total if max_pages is None else min(total, start_page - 1 + max_pages)
        for i in range(start_page - 1, last):
            raw = doc[i].get_text()
            cleaned = clean_text(raw)
            if cleaned:
                yield Page(number=i + 1, text=cleaned)
    finally:
        doc.close()


def pdf_page_count(path: str) -> int:
    import fitz

    doc = fitz.open(path)
    try:
        return doc.page_count
    finally:
        doc.close()
