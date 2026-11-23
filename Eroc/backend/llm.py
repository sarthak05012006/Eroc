"""Ollama client: structured-JSON chat, embeddings, warm-up and health. Nothing here calls a paid API."""
from __future__ import annotations

import json
import re
import threading
import time

import numpy as np
import requests

from . import config, logs
from .cache import LRU

log = logs.get("llm")

GATE = threading.Lock()          # one heavy model job at a time (the laptop can only run one anyway)
_session = requests.Session()    # keeps the connection open between calls
_query_cache = LRU(128)


class LLMError(RuntimeError):
    pass


class ModelMissing(LLMError):
    pass


def _post(path: str, payload: dict, timeout: int, retries: int | None = None) -> dict:
    """POST to Ollama. Retries connection errors and 5xx; fails fast on timeouts and missing models."""
    retries = config.LLM_RETRIES if retries is None else retries
    url = f"{config.OLLAMA_HOST}{path}"
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = _session.post(url, json=payload, timeout=timeout)
        except requests.Timeout as e:
            raise LLMError(f"Ollama timed out after {timeout}s on {path}. Try a smaller model or raise LLM_TIMEOUT.") from e
        except requests.ConnectionError as e:
            last = e
            log.warning("cannot reach Ollama at %s (attempt %d/%d)", config.OLLAMA_HOST, attempt + 1, retries + 1)
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 404 and "not found" in r.text.lower():
            raise ModelMissing(f"Model '{payload.get('model')}' is not installed in Ollama. Run: ollama pull {payload.get('model')}")
        if r.status_code >= 500:
            last = LLMError(f"Ollama error {r.status_code}: {r.text[:200]}")
            log.warning("%s (attempt %d/%d)", last, attempt + 1, retries + 1)
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise LLMError(f"Ollama rejected the request ({r.status_code}): {r.text[:200]}")
        return r.json()
    raise LLMError(f"Could not reach Ollama at {config.OLLAMA_HOST}: {last}")


def parse_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise LLMError("Model did not return valid JSON")


def chat_json(system: str, user: str, schema: dict | None = None, retries: int = 1, label: str = "chat",
              num_predict: int | None = None):
    """Deterministic (temperature 0) chat that must return JSON. `think:false` skips Qwen3.5's reasoning phase
    (much faster); the JSON schema keeps the output well-formed."""
    options = {"temperature": 0, "num_ctx": config.NUM_CTX}
    if num_predict:
        options["num_predict"] = num_predict
    payload = {"model": config.LLM_MODEL, "stream": False, "think": False, "keep_alive": config.KEEP_ALIVE,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "format": schema or "json", "options": options}
    last: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.time()
        data = _post("/api/chat", payload, config.LLM_TIMEOUT)
        secs = time.time() - t0
        n, dur = data.get("eval_count"), data.get("eval_duration")
        log.info("%s: %.1fs%s", label, secs, f" ({n / (dur / 1e9):.1f} tok/s)" if n and dur else "")
        try:
            return parse_json(data["message"]["content"])
        except (LLMError, KeyError) as e:
            last = e
            log.warning("%s: bad JSON on attempt %d (done_reason=%s)", label, attempt + 1, data.get("done_reason"))
    raise LLMError(str(last))


def embed(texts: list[str], batch: int = 16) -> np.ndarray:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        data = _post("/api/embed", {"model": config.EMBED_MODEL, "input": texts[i:i + batch],
                                    "keep_alive": config.KEEP_ALIVE}, 600)
        out.extend(data["embeddings"])
    arr = np.array(out, dtype=np.float32)
    return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)


def embed_query(text: str) -> np.ndarray:
    """Embedding of a single question, cached (repeat questions cost nothing)."""
    key = (config.EMBED_MODEL, text.strip().lower())
    hit = _query_cache.get(key)
    if hit is not None:
        return hit
    vec = embed([text])[0]
    _query_cache.set(key, vec)
    return vec


def warmup() -> None:
    """Load both models into memory so the first real question is not slow. Never raises."""
    try:
        t0 = time.time()
        _post("/api/generate", {"model": config.LLM_MODEL, "prompt": "", "keep_alive": config.KEEP_ALIVE}, 600, retries=0)
        _post("/api/embed", {"model": config.EMBED_MODEL, "input": ["warm-up"], "keep_alive": config.KEEP_ALIVE}, 600, retries=0)
        log.info("models warmed up in %.1fs", time.time() - t0)
    except LLMError as e:
        log.warning("warm-up skipped: %s", e)


def health() -> dict:
    try:
        t0 = time.time()
        r = _session.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        latency = round((time.time() - t0) * 1000)
        names = [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException as e:
        return {"ok": False, "error": str(e), "host": config.OLLAMA_HOST}
    try:
        loaded = [m["name"] for m in _session.get(f"{config.OLLAMA_HOST}/api/ps", timeout=5).json().get("models", [])]
    except (requests.RequestException, ValueError):
        loaded = []

    def has(want: str) -> bool:
        return want in names if ":" in want else any(n == want or n.split(":")[0] == want for n in names)

    return {"ok": True, "host": config.OLLAMA_HOST, "llm": config.LLM_MODEL, "embed": config.EMBED_MODEL,
            "llm_present": has(config.LLM_MODEL), "embed_present": has(config.EMBED_MODEL),
            "loaded": loaded, "latency_ms": latency, "models": names}
