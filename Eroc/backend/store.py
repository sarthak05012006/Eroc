"""Persistent index: documents, chunks, embeddings. Hybrid (dense + BM25, RRF) search."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

from . import logs
from .textutil import BM25, rrf, tokenize

log = logs.get("store")


def _atomic_write(path: Path, writer) -> None:
    """Write to a temp file, then rename: a crash mid-save can never leave a half-written index."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


class Store:
    def __init__(self, index_dir: Path):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.docs: dict[str, dict] = {}
        self.chunks: list[dict] = []
        self.emb = np.zeros((0, 1), dtype=np.float32)
        self.scan: dict = {"items": [], "finished": None}
        self._bm25: BM25 | None = None
        self._pos: dict[str, int] = {}
        self._by_doc_n: dict[tuple[str, int], int] = {}
        self.load()

    # ---------- persistence ----------
    def load(self):
        with self.lock:
            f = self.dir
            try:
                if (f / "docs.json").exists():
                    self.docs = json.loads((f / "docs.json").read_text(encoding="utf-8"))
                    self.chunks = json.loads((f / "chunks.json").read_text(encoding="utf-8"))
                    self.emb = np.load(f / "emb.npy") if (f / "emb.npy").exists() else np.zeros((0, 1), np.float32)
                    if len(self.emb) != len(self.chunks):
                        raise ValueError(f"{len(self.chunks)} chunks but {len(self.emb)} embeddings")
                if (f / "scan.json").exists():
                    self.scan = json.loads((f / "scan.json").read_text(encoding="utf-8"))
            except (ValueError, OSError) as e:
                log.error("index in %s is unreadable (%s); starting empty. Old files kept as *.bad", f, e)
                for name in ("docs.json", "chunks.json", "emb.npy", "scan.json"):
                    if (f / name).exists():
                        os.replace(f / name, f / (name + ".bad"))
                self.docs, self.chunks, self.emb = {}, [], np.zeros((0, 1), np.float32)
                self.scan = {"items": [], "finished": None}
            self._reindex()
            log.info("index loaded: %d documents, %d chunks", len(self.docs), len(self.chunks))

    def save(self):
        with self.lock:
            enc = lambda obj, **kw: (lambda fh: fh.write(json.dumps(obj, ensure_ascii=False, **kw).encode("utf-8")))
            _atomic_write(self.dir / "docs.json", enc(self.docs, indent=1))
            _atomic_write(self.dir / "chunks.json", enc(self.chunks))
            _atomic_write(self.dir / "emb.npy", lambda fh: np.save(fh, self.emb))
            _atomic_write(self.dir / "scan.json", enc(self.scan))

    @property
    def version(self) -> str:
        """Changes whenever documents, dates or amendment links change (used to invalidate cached answers)."""
        with self.lock:
            sig = json.dumps([[d["id"], d.get("sha"), d.get("doc_date"), d.get("effective_date"), sorted(d.get("amends", []))]
                              for d in sorted(self.docs.values(), key=lambda x: x["id"])])
        return hashlib.sha1(sig.encode()).hexdigest()[:12]

    def stats(self) -> dict:
        return {"documents": len(self.docs), "chunks": len(self.chunks), "pages": sum(d.get("pages", 0) for d in self.docs.values()),
                "index_version": self.version}

    def _reindex(self):
        self._pos = {c["id"]: i for i, c in enumerate(self.chunks)}
        self._by_doc_n = {(c["doc_id"], c["n"]): i for i, c in enumerate(self.chunks)}
        self._bm25 = BM25([tokenize(c["section"] + " " + c["text"]) for c in self.chunks]) if self.chunks else None

    # ---------- documents ----------
    def doc_by_sha(self, sha: str):
        return next((d for d in self.docs.values() if d.get("sha") == sha), None)

    def add_document(self, doc: dict, chunks: list[dict], embs: np.ndarray):
        with self.lock:
            self.docs[doc["id"]] = doc
            self.chunks.extend(chunks)
            self.emb = embs if len(self.emb) == 0 else np.vstack([self.emb, embs])
            self._reindex()
            self.save()

    def remove_document(self, doc_id: str):
        with self.lock:
            self.docs.pop(doc_id, None)
            keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
            self.chunks = [self.chunks[i] for i in keep]
            self.emb = self.emb[keep] if keep else np.zeros((0, 1), np.float32)
            for d in self.docs.values():
                d["amends"] = [a for a in d.get("amends", []) if a != doc_id]
            self.scan = {"items": [], "finished": None}
            self._reindex()
            self.save()

    def doc_chunks(self, doc_id: str) -> list[dict]:
        return [c for c in self.chunks if c["doc_id"] == doc_id]

    def chunk(self, chunk_id: str):
        i = self._pos.get(chunk_id)
        return self.chunks[i] if i is not None else None

    def neighbor(self, chunk: dict, delta: int):
        i = self._by_doc_n.get((chunk["doc_id"], chunk["n"] + delta))
        return self.chunks[i] if i is not None else None

    # ---------- search ----------
    def search(self, qvec: np.ndarray, qtext: str, k: int, doc_ids: list[str] | None = None) -> list[dict]:
        with self.lock:
            if not self.chunks or self._bm25 is None:
                return []
            sims = self.emb @ qvec
            bm = self._bm25.scores(tokenize(qtext, drop_stop=True))
            allowed = np.ones(len(self.chunks), dtype=bool)
            if doc_ids:
                allowed = np.array([c["doc_id"] in set(doc_ids) for c in self.chunks])
            dense = [int(i) for i in np.argsort(-np.where(allowed, sims, -9))[:50] if allowed[i]]
            lex = [int(i) for i in np.argsort(-np.where(allowed, bm, -9))[:50] if allowed[i] and bm[i] > 0]
            fused = rrf([dense, lex])
            top = sorted(fused, key=fused.get, reverse=True)[:k]
            return [{"chunk": self.chunks[i], "sim": float(sims[i]), "bm25": float(bm[i]), "score": fused[i]} for i in top]

    def best_chunks_for_docs(self, qvec: np.ndarray, doc_ids: list[str]) -> dict[str, tuple[dict, float]]:
        """For each document, its chunk most similar to the question (used to pull in every version of a rule)."""
        with self.lock:
            if not self.chunks:
                return {}
            sims = self.emb @ qvec
            best: dict[str, tuple[dict, float]] = {}
            wanted = set(doc_ids)
            for i, c in enumerate(self.chunks):
                d = c["doc_id"]
                if d in wanted and (d not in best or sims[i] > best[d][1]):
                    best[d] = (c, float(sims[i]))
            return best
