from fastapi import APIRouter

from .. import config, llm
from ..state import store

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    h = llm.health()
    h.update(store.stats())
    h["app_version"] = config.VERSION
    return h


@router.get("/config")
def get_config():
    return config.summary()
