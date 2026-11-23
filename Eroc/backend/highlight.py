"""Render a PDF page to PNG with the cited passage highlighted."""
from __future__ import annotations

from pathlib import Path


def _rects(page, quote: str):
    quote = " ".join(quote.split())
    found = page.search_for(quote)
    if found:
        return found
    parts = [p.strip() for p in quote.split(" | ") if len(p.strip()) > 2]        # table rows
    if len(parts) > 1:
        found = [r for p in parts for r in page.search_for(p)]
        if found:
            return found
    words = quote.split()
    found = []
    for i in range(0, len(words), 5):                                           # sliding 5-word windows
        found += page.search_for(" ".join(words[i:i + 5]))
    return found


def render_page(pdf_path: Path, page_no: int, quote: str | None = None, dpi: int = 110) -> bytes:
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        if not 1 <= page_no <= len(doc):
            raise IndexError("page out of range")
        page = doc[page_no - 1]
        if quote:
            for r in _rects(page, quote):
                page.draw_rect(r, color=None, fill=(1, 0.85, 0.1), fill_opacity=0.45)
        return page.get_pixmap(dpi=dpi).tobytes("png")
