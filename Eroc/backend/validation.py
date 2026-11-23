"""Input checks shared by the API routes. Pure Python, so they are easy to unit-test."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from . import config


class BadRequest(ValueError):
    """Turned into HTTP 400 by the app."""


def check_question(q: str | None) -> str:
    q = (q or "").strip()
    if not q:
        raise BadRequest("Please type a question.")
    if len(q) > config.MAX_QUESTION_CHARS:
        raise BadRequest(f"Question is too long ({len(q)} characters, limit {config.MAX_QUESTION_CHARS}).")
    return q


def check_as_of(value: str | None) -> str | None:
    if not value:
        return None
    try:
        date.fromisoformat(value)
    except ValueError:
        raise BadRequest(f"as_of must be a date like 2026-11-23, got '{value}'.")
    return value


def check_doc_ids(ids: list[str] | None, known: dict) -> list[str] | None:
    if not ids:
        return None
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise BadRequest(f"Unknown document id(s): {', '.join(unknown)}")
    return ids


def check_iso_date(value: str | None, field: str) -> str | None:
    if value in (None, ""):
        return None
    try:
        date.fromisoformat(value)
    except ValueError:
        raise BadRequest(f"{field} must be YYYY-MM-DD, got '{value}'.")
    return value


def safe_name(filename: str | None) -> str:
    """Strip any path and odd characters from an uploaded file name (blocks '../' tricks)."""
    name = Path((filename or "document.pdf").replace("\\", "/")).name
    name = re.sub(r"[^\w.\- ()\u0900-\u097F]", "_", name).strip(" .") or "document.pdf"
    return name[:150]


def check_pdf_file(path: Path, name: str) -> None:
    if not name.lower().endswith(".pdf"):
        raise BadRequest(f"{name}: only PDF files are supported.")
    with path.open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise BadRequest(f"{name}: this does not look like a real PDF file.")
