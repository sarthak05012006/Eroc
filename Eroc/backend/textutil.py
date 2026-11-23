"""Pure-python text helpers: normalisation, verbatim-quote matching, BM25, number guard."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import numpy as np

_WORD = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_STOP = set("the a an of to in on for and or is are was were be been by with as at from that this it its "
            "what which who whom how when where why do does did can could should would shall will may "
            "any all if than then there their they them has have had not no".split())


def ascii_digits(s: str) -> str:
    """Map any Unicode decimal digit (e.g. Devanagari) to ASCII."""
    return "".join(str(unicodedata.digit(ch)) if unicodedata.category(ch) == "Nd" else ch for ch in s)


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")
    s = re.sub(r"-\s*\n\s*(?=[a-z])", "", s)
    return s


def tokenize(s: str, drop_stop: bool = False) -> list[str]:
    toks = [t.casefold() for t in _WORD.findall(norm_text(s))]
    return [t for t in toks if t not in _STOP] if drop_stop else toks


def _spans(text: str):
    return [(m.group().casefold(), m.start(), m.end()) for m in _WORD.finditer(text)]


MIN_QUOTE_TOKENS = 3


def find_quote(quote: str, text: str, min_ratio: float = 0.9):
    """Locate `quote` inside `text`.

    Returns (score, snapped, (start, end)). `snapped` is the exact substring of `text`
    that best matches, so whatever we display is guaranteed verbatim.
    """
    q = [t for t, _, _ in _spans(norm_text(quote))]
    spans = _spans(text)
    T = [t for t, _, _ in spans]
    n = len(q)
    if n < MIN_QUOTE_TOKENS or len(T) < 1:
        return 0.0, "", None
    for i in range(len(T) - n + 1):               # exact token-sequence match
        if T[i:i + n] == q:
            a, b = spans[i][1], spans[i + n - 1][2]
            return 1.0, text[a:b], (a, b)
    qset = set(q)
    best, best_i, best_size = 0.0, -1, 0
    for size in sorted({max(MIN_QUOTE_TOKENS, n - 2), n, n + 2}):
        for i in range(0, max(1, len(T) - size + 1)):
            w = T[i:i + size]
            if len(qset & set(w)) < 0.6 * len(qset):
                continue
            r = SequenceMatcher(None, q, w, autojunk=False).ratio()
            if r > best:
                best, best_i, best_size = r, i, len(w)
    if best_i >= 0 and best >= min_ratio:
        a, b = spans[best_i][1], spans[best_i + best_size - 1][2]
        return best, text[a:b], (a, b)
    return best, "", None


def canon_num(raw: str) -> str:
    raw = raw.replace(",", "")
    try:
        return f"{float(raw):g}"
    except ValueError:
        return raw


def numbers(s: str) -> set[str]:
    return {canon_num(m) for m in _NUM.findall(ascii_digits(norm_text(s)))}


def unsupported_numbers(claim: str, evidence: list[str], extra_allowed: list[str] = ()) -> list[str]:
    """Numbers stated in `claim` that appear nowhere in the quoted evidence.
    Small bare integers (<=10) are ignored so 'two conditions' does not trip the guard."""
    allowed = set()
    for e in list(evidence) + list(extra_allowed):
        allowed |= numbers(e)
    claim = ascii_digits(norm_text(claim))
    bad = []
    for m in _NUM.finditer(claim):
        c = canon_num(m.group())
        if c in allowed:
            continue
        nxt = claim[m.end():m.end() + 1]
        if float(c) <= 10 and "." not in c and nxt != "%":
            continue
        bad.append(m.group())
    return bad


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.len = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs_tokens) else 0.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(q_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(t)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1)))
        return out


def rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return fused
