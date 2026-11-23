"""Claim-check mode: the user states a fact; Veritas answers Supported / Contradicted / Partial / Not found.

Same zero-guessing rules as Q&A: a quote must exist verbatim in a PDF (checked in code), every figure in the claim must
appear in the quotes (checked in code), and a strict judge must agree. Anything else is "not_found"."""
from __future__ import annotations

import json
import re
import time
from datetime import date

from . import config, llm, logs, qa
from .textutil import ascii_digits, find_quote, norm_text, unsupported_numbers

log = logs.get("claimcheck")
MAX_CLAIMS = 10

CHECK_SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["supported", "contradicted", "not_found"]},
    "reason": {"type": "string"},
    "evidence": {"type": "array", "items": {"type": "object", "properties": {
        "chunk": {"type": "string"}, "quote": {"type": "string"}}, "required": ["chunk", "quote"]}}},
    "required": ["verdict", "reason", "evidence"]}

CHECK_SYSTEM = """You verify ONE claim against numbered EVIDENCE passages from official documents.
RULES
1. Use ONLY the EVIDENCE. Never use outside knowledge. Never guess.
2. verdict "supported": the evidence states the claim (names, numbers, dates and conditions all match).
   verdict "contradicted": the evidence states something different about the same fact (e.g. another date or name).
   verdict "not_found": the evidence says nothing about this fact.
3. For supported or contradicted, give 1-3 evidence items {"chunk": "C2", "quote": "..."}; each quote must be COPIED EXACTLY,
   character for character, from that passage. For not_found, evidence is [].
4. Spelling variants of the same name in Hindi and English, or different date formats of the same date, count as a match.
5. "reason" is one short plain sentence in the language of the claim.
Return JSON {"verdict","reason","evidence":[{"chunk","quote"}]}"""


_DATE = re.compile(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?!\d)|(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")


def dates_in(text: str) -> set[tuple[int, int, int]]:
    """Calendar dates as (year, month, day) so 09-07-2010, 9/7/2010 and 2010-07-09 compare equal."""
    out = set()
    for m in _DATE.finditer(ascii_digits(norm_text(text))):
        dd, mm, yyyy, y, mo, d = m.groups()                    # first form: dd-mm-yyyy, second: yyyy-mm-dd
        out.add((int(yyyy), int(mm), int(dd)) if dd else (int(y), int(mo), int(d)))
    return out


def split_claims(text: str | list[str]) -> list[str]:
    """One claim per line or per ';'. Blank lines are dropped."""
    parts = text if isinstance(text, list) else re.split(r"[\n;]+", text or "")
    claims = [p.strip(" \t-•*") for p in parts if p and p.strip(" \t-•*")]
    if not claims:
        raise ValueError("Please enter a claim to check.")
    if len(claims) > MAX_CLAIMS:
        raise ValueError(f"Too many claims ({len(claims)}); the limit is {MAX_CLAIMS} per check.")
    if any(len(c) > config.MAX_QUESTION_CHARS for c in claims):
        raise ValueError("A claim is too long.")
    return claims


def _result(claim, verdict, reason, cites=None, best=0.0, secs=0.0):
    return {"claim": claim, "verdict": verdict, "reason": reason, "citations": cites or [],
            "best_sim": round(best, 3), "seconds": round(secs, 1)}


def check_one(store, claim: str, as_of: str, doc_ids=None) -> dict:
    t0 = time.time()
    qv = llm.embed_query(claim)
    hits = store.search(qv, claim, config.TOP_K, doc_ids)
    best = max((h["sim"] for h in hits), default=0.0)
    if not hits or best < config.MIN_SIM:
        return _result(claim, "not_found", "No passage in your documents is about this, so there is no proof either way.",
                       best=best, secs=time.time() - t0)

    evidence = qa.build_evidence(store, hits, qv)
    evid = {e["key"]: e for e in evidence}
    ctx = "\n\n".join(f'[{e["key"]}] {e["chunk"]["doc"]}, page {e["chunk"]["page"]}\n{e["chunk"]["text"]}' for e in evidence)
    draft = llm.chat_json(CHECK_SYSTEM, f"CLAIM: {claim}\n\nEVIDENCE:\n{ctx}", CHECK_SCHEMA, label="claimcheck")
    verdict = draft.get("verdict") if draft.get("verdict") in ("supported", "contradicted", "not_found") else "not_found"
    reason = str(draft.get("reason") or "").strip()

    # 1. code check: every quote must exist verbatim in the cited passage
    cites = []
    for ev in (draft.get("evidence") or [])[:3]:
        item = evid.get(ev.get("chunk"))
        if not item:
            continue
        score, snapped, _ = find_quote(ev.get("quote", ""), item["chunk"]["text"], config.QUOTE_MIN_RATIO)
        if snapped:
            cites.append(qa._cite(item, snapped, score))
    if verdict != "not_found" and not cites:
        return _result(claim, "not_found", "The model's quote could not be found word-for-word in the PDF, so it was rejected.",
                       best=best, secs=time.time() - t0)

    if verdict == "supported":
        # 2. code check: every figure in the claim must be in the quotes
        quotes = [c["quote"] for c in cites]
        qdates = set().union(*(dates_in(q) for q in quotes))
        wrong = [d for d in dates_in(claim) if d not in qdates]
        if wrong and qdates:
            shown = ", ".join(sorted(f"{d[2]:02d}-{d[1]:02d}-{d[0]}" for d in qdates))
            return _result(claim, "contradicted", f"The document gives a different date: {shown}.", cites, best, time.time() - t0)
        bad = unsupported_numbers(claim, quotes, [as_of]) + [f"{d[2]:02d}-{d[1]:02d}-{d[0]}" for d in wrong]
        if bad:
            return _result(claim, "partial", f'The documents back part of this, but not the figure(s): {", ".join(bad)}.',
                           cites, best, time.time() - t0)
        # 3. strict judge must agree (fails closed)
        try:
            res = llm.chat_json(qa.JUDGE_SYSTEM, json.dumps([{"id": 0, "claim": claim, "quotes": [c["quote"] for c in cites]}],
                                                            ensure_ascii=False), qa.JUDGE_SCHEMA, label="claimjudge")
            v = next((r for r in res.get("results", []) if isinstance(r, dict) and r.get("id") == 0), {})
        except llm.LLMError as e:
            log.error("judge unavailable: %s", e)
            return _result(claim, "not_found", "The verifier model was unreachable, so this was not confirmed.",
                           cites, best, time.time() - t0)
        jv = v.get("verdict")
        if jv == "contradicted":
            verdict, reason = "contradicted", v.get("reason") or reason
        elif jv != "supported":
            verdict, reason = ("partial", v.get("reason") or "Only part of the claim is backed by the quote.")
    log.info("CLAIM %s: %s (best_sim=%.3f)", verdict, claim[:60], best)
    return _result(claim, verdict, reason, cites, best, time.time() - t0)


def check_claims(store, text, as_of: str | None = None, doc_ids=None) -> dict:
    as_of = as_of or date.today().isoformat()
    date.fromisoformat(as_of)
    claims = split_claims(text)
    if not store.docs:
        return {"as_of": as_of, "results": [_result(c, "not_found", "No documents have been uploaded yet.") for c in claims],
                "summary": {"supported": 0, "contradicted": 0, "partial": 0, "not_found": len(claims)}}
    results = [check_one(store, c, as_of, doc_ids) for c in claims]
    summary = {k: sum(r["verdict"] == k for r in results) for k in ("supported", "contradicted", "partial", "not_found")}
    return {"as_of": as_of, "results": results, "summary": summary}