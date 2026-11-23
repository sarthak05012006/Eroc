from datetime import datetime, timezone

from fastapi import APIRouter

from .. import llm, reasoning
from ..state import jobs, store

router = APIRouter(prefix="/api", tags=["scan"])
IDLE = {"state": "idle", "running": False, "done": 0, "total": 0, "message": "", "errors": [], "error_count": 0, "error": None}


def _scan(job):
    items, errors = reasoning.scan_all(store, progress=lambda d, t: job.progress(d, t, f"scanning group {d} of {t}"),
                                       lock=llm.GATE)
    job.errors.extend(errors)
    store.scan = {"items": items, "finished": datetime.now(timezone.utc).isoformat(), "errors": errors[:10]}
    store.save()


@router.post("/scan")
def start_scan():
    return jobs.start("scan", _scan).to_dict()


@router.get("/scan")
def scan_status():
    job = jobs.latest("scan")
    return {"job": job.to_dict() if job else IDLE, "result": store.scan}


@router.get("/chain")
def chain():
    from datetime import date
    return reasoning.chain_view(store, list(store.docs), date.today().isoformat())
