import json
import queue
import threading

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import claimcheck, config, llm, logs, qa, validation
from ..state import store

router = APIRouter(prefix="/api", tags=["ask"])
log = logs.get("ask")


class AskBody(BaseModel):
    question: str
    as_of: str | None = None
    doc_ids: list[str] | None = None
    check_conflicts: bool = True


def _checked(body: AskBody):
    return (validation.check_question(body.question), validation.check_as_of(body.as_of),
            validation.check_doc_ids(body.doc_ids, store.docs))


@router.post("/ask")
def ask(body: AskBody):
    q, as_of, doc_ids = _checked(body)
    with llm.GATE:
        return qa.ask(store, q, as_of, doc_ids, body.check_conflicts)


@router.post("/ask/stream")
def ask_stream(body: AskBody):
    """Same as /ask, but streams progress (Server-Sent Events): `stage` events, then one `result` (or `error`)."""
    q, as_of, doc_ids = _checked(body)
    events: queue.Queue = queue.Queue()

    def worker():
        try:
            events.put(("stage", {"stage": "queued"}))
            with llm.GATE:
                result = qa.ask(store, q, as_of, doc_ids, body.check_conflicts,
                                on_stage=lambda name, **info: events.put(("stage", {"stage": name, **info})))
            events.put(("result", result))
        except (llm.LLMError, ValueError) as e:
            log.warning("ask failed: %s", e)
            events.put(("error", {"message": str(e)}))
        except Exception as e:
            log.exception("ask crashed")
            events.put(("error", {"message": f"Internal error: {e}"}))
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True, name="ask").start()

    def stream():
        while (item := events.get()) is not None:
            name, data = item
            yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/retrieve")
def retrieve(body: AskBody):
    """Debug view: what the search finds for a question (no model call except the embedding). Use it to tune MIN_SIM."""
    q, _, doc_ids = _checked(body)
    hits = store.search(llm.embed_query(q), q, config.TOP_K, doc_ids)
    return {"min_sim": config.MIN_SIM, "best_sim": round(max((h["sim"] for h in hits), default=0.0), 3),
            "hits": [{"doc": h["chunk"]["doc"], "page": h["chunk"]["page"], "kind": h["chunk"]["kind"],
                      "sim": round(h["sim"], 3), "bm25": round(h["bm25"], 2), "text": h["chunk"]["text"][:240]} for h in hits]}


class ClaimBody(BaseModel):
    claims: str | list[str]
    as_of: str | None = None
    doc_ids: list[str] | None = None


@router.post("/claim-check")
def claim_check(body: ClaimBody):
    """Check stated facts (one per line) against the documents: supported / contradicted / partial / not_found."""
    as_of = validation.check_as_of(body.as_of)
    doc_ids = validation.check_doc_ids(body.doc_ids, store.docs)
    try:
        with llm.GATE:
            return claimcheck.check_claims(store, body.claims, as_of, doc_ids)
    except ValueError as e:
        raise validation.BadRequest(str(e))