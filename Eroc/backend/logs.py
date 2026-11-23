"""One logger setup for the whole app. Use:  log = logs.get("qa");  log.info(...)"""
import logging
import sys

_ready = False


def setup(level: str = "INFO") -> None:
    global _ready
    if _ready:
        return
    root = logging.getLogger("veritas")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)-14s %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    _ready = True


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"veritas.{name}")
