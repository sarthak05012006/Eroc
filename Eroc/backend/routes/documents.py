import uuid
from datetime import date

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .. import config, ingest, llm, logs, reasoning, validation
from ..state import store

router = APIRouter(prefix="/api", tags=["documents"])
log = logs.get("documents")


class DocPatch(BaseModel):
    doc_date: str | None = None
    effective_date: str | None = None
    amends: list[str] | None = None


def public(d: dict) -> dict:
    d = {k: v for k, v in d.items() if k != "link_checked"}
    return {**d, "status": reasoning.status_of(d, date.today().isoformat())}


def _doc_or_404(doc_id: str) -> dict:
    d = store.docs.get(doc_id)
    if not d:
        raise HTTPException(404, "unknown document")
    return d


@router.get("/documents")
def list_documents():
    return sorted((public(d) for d in store.docs.values()), key=lambda d: reasoning.eff_date(d))


@router.post("/documents")
def upload(files: list[UploadFile] = File(...)):
    """Index PDFs. Failures are reported per file in the response (the HTTP status is still 200)."""
    results, limit = [], config.MAX_UPLOAD_MB * 1024 * 1024
    with llm.GATE:
        for f in files:
            name = validation.safe_name(f.filename)
            tmp = config.PDF_DIR / f"_upload_{uuid.uuid4().hex}.pdf"
            try:
                size = 0
                with tmp.open("wb") as out:
                    while block := f.file.read(1024 * 1024):
                        size += len(block)
                        if size > limit:
                            raise validation.BadRequest(f"{name}: larger than {config.MAX_UPLOAD_MB} MB.")
                        out.write(block)
                validation.check_pdf_file(tmp, name)
                doc, new = ingest.ingest_pdf(tmp, store, llm.embed, display_name=name)
                if new:
                    tmp.replace(config.PDF_DIR / f"{doc['id']}.pdf")
                log.info("indexed %s: %d pages, %d chunks, %d OCR pages%s", name, doc["pages"], doc["n_chunks"],
                         doc["ocr_pages"], "" if new else " (already indexed)")
                results.append({"name": name, "id": doc["id"], "new": new, "pages": doc["pages"], "ocr_pages": doc["ocr_pages"]})
            except (validation.BadRequest, ValueError, llm.LLMError) as e:
                log.warning("upload of %s rejected: %s", name, e)
                results.append({"name": name, "error": str(e)})
            except Exception as e:
                log.exception("unexpected error while indexing %s", name)
                results.append({"name": name, "error": f"Unexpected error: {e}"})
            finally:
                tmp.unlink(missing_ok=True)
        if any(r.get("new") for r in results):
            try:
                calls = reasoning.link_amendments(store)
                log.info("amendment linking done (%d model calls)", calls)
            except Exception as e:
                log.exception("amendment linking failed")
                results.append({"name": "(amendment linking)", "error": str(e)})
            store.scan = {"items": [], "finished": None}
            store.save()
    return results


@router.patch("/documents/{doc_id}")
def patch_document(doc_id: str, body: DocPatch):
    d = _doc_or_404(doc_id)
    if body.doc_date is not None:
        d["doc_date"] = validation.check_iso_date(body.doc_date, "doc_date")
    if body.effective_date is not None:
        d["effective_date"] = validation.check_iso_date(body.effective_date, "effective_date")
    if body.amends is not None:
        d["amends"] = [a for a in body.amends if a in store.docs and a != doc_id]
        d["amends_locked"] = True
    store.scan = {"items": [], "finished": None}
    store.save()
    return public(d)


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    _doc_or_404(doc_id)
    store.remove_document(doc_id)
    (config.PDF_DIR / f"{doc_id}.pdf").unlink(missing_ok=True)
    return {"ok": True}


@router.get("/documents/{doc_id}/pdf")
def get_pdf(doc_id: str):
    _doc_or_404(doc_id)
    p = config.PDF_DIR / f"{doc_id}.pdf"
    if not p.exists():
        raise HTTPException(404, "file missing")
    return FileResponse(p, media_type="application/pdf")


@router.get("/documents/{doc_id}/page/{page}.png")
def page_png(doc_id: str, page: int, quote: str | None = None):
    from ..highlight import render_page
    _doc_or_404(doc_id)
    p = config.PDF_DIR / f"{doc_id}.pdf"
    if not p.exists():
        raise HTTPException(404, "file missing")
    try:
        return Response(render_page(p, page, quote), media_type="image/png")
    except IndexError:
        raise HTTPException(404, "page out of range")


@router.get("/documents/{doc_id}/chunks")
def get_chunks(doc_id: str):
    """Debug view: exactly how this document was split (page, kind, section, text)."""
    _doc_or_404(doc_id)
    return [{k: c[k] for k in ("id", "page", "kind", "section", "text")} for c in store.doc_chunks(doc_id)]
