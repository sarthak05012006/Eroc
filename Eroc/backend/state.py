"""Shared singletons: the document store and the background-job registry."""
from . import config
from .jobs import JobRegistry
from .store import Store

config.PDF_DIR.mkdir(parents=True, exist_ok=True)
store = Store(config.INDEX_DIR)
jobs = JobRegistry()
