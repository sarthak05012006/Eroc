"""PDF -> pages (text layer, tables, OCR fallback) -> chunks with page numbers -> index."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from . import config

MIN_TEXT_CHARS = 40

# ------------------------------------------------------------------ dates
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_DATE_RES = [
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}\.?,?\s+(\d{{4}})\b", re.I), "dmy"),
    (re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I), "mdy"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    (re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b"), "num"),
]


def _iso(y, m, d):
    try:
        return datetime(int(y), int(m), int(d)).date().isoformat()
    except ValueError:
        return None


def find_dates(text: str) -> list[tuple[str, int, int]]:
    found = []
    for rx, kind in _DATE_RES:
        for m in rx.finditer(text):
            g = m.groups()
            if kind == "dmy":
                iso = _iso(g[2], _MONTHS[g[1][:3].lower()], g[0])
            elif kind == "mdy":
                iso = _iso(g[2], _MONTHS[g[0][:3].lower()], g[1])
            elif kind == "iso":
                iso = _iso(*g)
            else:
                iso = _iso(g[2], g[1], g[0])          # Indian dd/mm/yyyy
            if iso:
                found.append((iso, m.start(), m.end()))
    found.sort(key=lambda x: x[1])
    out, last_end = [], -1
    for f in found:                                    # drop overlapping matches
        if f[1] >= last_end:
            out.append(f)
            last_end = f[2]
    return out


_EFF = re.compile(r"(effective|w\.e\.f|with effect from|come into (?:force|effect)|applicable from|in force from)", re.I)
_DOCD = re.compile(r"(dated|date|issued|executed|made on|signed)", re.I)


def guess_dates(text: str) -> tuple[str | None, str | None]:
    """Return (document date, effective date) guessed from the first pages."""
    dates = find_dates(text)
    doc_date = eff = None
    for iso, s, _ in dates:
        before = text[max(0, s - 45):s]
        if eff is None and _EFF.search(before):
            eff = iso
        elif doc_date is None and _DOCD.search(before):
            doc_date = iso
    if doc_date is None:
        doc_date = next((d[0] for d in dates if d[0] != eff), eff)
    return doc_date, (eff or doc_date)


# ------------------------------------------------------------------ parsing
def _ocr(page) -> str:
    """OCR one page. Tries several settings and keeps the best result. Logs the real reason if it fails."""
    import io
    import logging
    log = logging.getLogger("veritas.ocr")
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except Exception as e:
        log.error("OCR packages missing (pip install pytesseract pillow): %s", e)
        return ""
    try:
        langs = pytesseract.get_languages(config="")
    except Exception as e:
        log.error("Tesseract program not found or not working: %s", e)
        return ""
    lang = "eng+hin" if "hin" in langs else "eng"

    best = ""
    for dpi in (300, 400):
        try:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
            img = ImageOps.autocontrast(img)
            try:  # straighten sideways / upside-down photos
                osd = pytesseract.image_to_osd(img, config="--psm 0")
                m = re.search(r"Rotate:\s*(\d+)", osd)
                if m and int(m.group(1)):
                    img = img.rotate(-int(m.group(1)), expand=True, fillcolor=255)
            except Exception:
                pass
            for psm in (3, 6, 11):
                txt = pytesseract.image_to_string(img, lang=lang, config=f"--psm {psm}")
                if len(txt.strip()) > len(best.strip()):
                    best = txt
                if len(best.strip()) >= MIN_TEXT_CHARS:
                    return best
        except Exception as e:
            log.warning("OCR attempt failed (dpi=%s): %s", dpi, e)
    if not best.strip():
        log.warning("OCR found no readable text on this page (blank, handwriting, or very low quality image)")
    return best

def _table_text(rows) -> str:
    lines = []
    for r in rows:
        cells = [(c or "").replace("\n", " ").strip() for c in r]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_pdf(path: Path) -> list[dict]:
    """Per page: reading-order text blocks (tables removed), tables as 'a | b | c' rows, OCR if no text layer."""
    import pymupdf
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, 1):
            tables, rects = [], []
            try:
                for t in page.find_tables().tables:
                    txt = _table_text(t.extract())
                    if txt:
                        tables.append(txt)
                        rects.append(pymupdf.Rect(t.bbox))
            except Exception:
                pass
            blocks = []
            for b in page.get_text("blocks", sort=True):
                if b[6] != 0:
                    continue
                r = pymupdf.Rect(b[:4])
                area = r.get_area()
                if area and any((r & tr).get_area() > 0.5 * area for tr in rects):
                    continue
                blocks.append(b[4])
            method = "text"
            if len("".join(blocks)) + sum(map(len, tables)) < MIN_TEXT_CHARS:
                ocr = _ocr(page)
                if ocr.strip():
                    blocks, method = [ocr], "ocr"
            pages.append({"page": i, "text": "\n\n".join(blocks), "tables": tables, "method": method})
    return pages


# ------------------------------------------------------------------ chunking
_HEAD = re.compile(r"^(?:(?:section|clause|article|schedule|annexure|part|chapter)\s+[\w.]+|\d+(?:\.\d+)*[.)]?\s+\S)", re.I)


def clean_para(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).replace("\u00ad", "")
    t = re.sub(r"-\s*\n\s*(?=[a-z])", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _split_long(p: str) -> list[str]:
    if len(p) <= config.CHUNK_CHARS:
        return [p]
    out, cur = [], ""
    for s in re.split(r"(?<=[.;!?])\s+", p):
        if cur and len(cur) + len(s) + 1 > config.CHUNK_CHARS:
            out.append(cur)
            cur = ""
        cur = f"{cur} {s}".strip()
    return out + ([cur] if cur else [])


def make_chunks(doc_id: str, name: str, pages: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    section = ""

    def add(text, page, kind, sec):
        chunks.append({"id": f"{doc_id}:{len(chunks)}", "doc_id": doc_id, "doc": name, "n": len(chunks),
                       "page": page, "kind": kind, "section": sec, "text": text})

    for p in pages:
        cur, cur_sec = "", section
        for raw in p["text"].split("\n\n"):
            para = clean_para(raw)
            if not para:
                continue
            if _HEAD.match(para) and len(para) < 120:
                section = para
            for piece in _split_long(para):
                if cur and len(cur) + len(piece) + 1 > config.CHUNK_CHARS:
                    add(cur, p["page"], "text", cur_sec)
                    cur = ""
                if not cur:
                    cur_sec = section
                cur = f"{cur} {piece}".strip()
        if cur:
            add(cur, p["page"], "text", cur_sec)
        for t in p.get("tables", []):
            add(unicodedata.normalize("NFKC", t), p["page"], "table", section)
    return chunks


# ------------------------------------------------------------------ ingest
def ingest_pdf(path: Path, store, embed_fn, display_name: str | None = None):
    """Returns (doc_record, is_new)."""
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    existing = store.doc_by_sha(sha)
    if existing:
        return existing, False
    pages = parse_pdf(path)
    if not any(p["text"].strip() or p["tables"] for p in pages):
        raise ValueError("No extractable text found (is this an image-only PDF and is Tesseract installed?)")
    doc_id = sha[:10]
    name = display_name or path.name
    chunks = make_chunks(doc_id, name, pages)
    first = "\n".join(p["text"] for p in pages[:2])
    doc_date, eff = guess_dates(first)
    embs = embed_fn([(c["section"] + "\n" + c["text"]).strip() for c in chunks])
    doc = {"id": doc_id, "name": name, "sha": sha, "pages": len(pages), "n_chunks": len(chunks),
           "ocr_pages": sum(p["method"] == "ocr" for p in pages), "doc_date": doc_date, "effective_date": eff,
           "amends": [], "amends_locked": False, "added": datetime.now(timezone.utc).isoformat()}
    store.add_document(doc, chunks, embs)
    return doc, True
