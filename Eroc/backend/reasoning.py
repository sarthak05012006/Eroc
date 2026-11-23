"""Amendment chains, contradiction detection, and 'which version applies today'.

LLM calls only *propose* (links, conflicting passage pairs). Every proposal is then checked
by code: quotes must exist verbatim in the cited chunk, and the amendment/date logic that
decides which statement prevails is deterministic."""
from __future__ import annotations

import contextlib
import re
from datetime import date

import numpy as np

from . import config, llm, logs
from .textutil import find_quote, numbers

log = logs.get("reasoning")

FAR = "9999-12-31"
_AMEND_WORDS = re.compile(r"amend|supersed|modif|replac|substitut|in lieu of|in partial modification", re.I)


# ------------------------------------------------------------------ document graph
def eff_date(doc: dict) -> str:
    return doc.get("effective_date") or doc.get("doc_date") or "0000-00-00"


def status_of(doc: dict, as_of: str) -> str:
    return "not yet in force" if eff_date(doc) > as_of else "in force"


def _undirected(docs: dict) -> dict[str, set[str]]:
    adj = {i: set() for i in docs}
    for d in docs.values():
        for t in d.get("amends", []):
            if t in docs:
                adj[d["id"]].add(t)
                adj[t].add(d["id"])
    return adj


def components(docs: dict) -> list[list[str]]:
    adj, seen, out = _undirected(docs), set(), []
    for start in docs:
        if start in seen:
            continue
        comp, stack = [], [start]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        out.append(comp)
    return out


def same_chain(docs: dict, a: str, b: str) -> bool:
    return any(a in c and b in c for c in components(docs))


def classify(docs: dict, a_id: str, b_id: str, as_of: str) -> dict:
    """Decide how two conflicting statements relate."""
    a, b = docs[a_id], docs[b_id]
    if a_id == b_id:
        return {"status": "contradiction", "winner": None, "note": "Two statements inside the same document disagree."}
    a_pending, b_pending = eff_date(a) > as_of, eff_date(b) > as_of
    if same_chain(docs, a_id, b_id):
        if a_pending != b_pending:
            pend, cur = (a, b) if a_pending else (b, a)
            return {"status": "amendment_pending", "winner": "b" if a_pending else "a",
                    "note": f"{pend['name']} changes this from {eff_date(pend)}; until then {cur['name']} applies."}
        if eff_date(a) != eff_date(b):
            win = "a" if eff_date(a) > eff_date(b) else "b"
        elif b_id in a.get("amends", []):
            win = "a"
        elif a_id in b.get("amends", []):
            win = "b"
        else:
            return {"status": "contradiction", "winner": None, "note": "Linked documents, but their dates do not settle which prevails."}
        w = a if win == "a" else b
        l = b if win == "a" else a
        return {"status": "superseded", "winner": win,
                "note": f"{w['name']} (effective {eff_date(w)}) amends {l['name']} (effective {eff_date(l)}), so {w['name']} applies."}
    return {"status": "contradiction", "winner": None,
            "note": "These documents are not linked by an amendment, so neither can be treated as prevailing."}


def chain_view(store, doc_ids: list[str], as_of: str, citations: list[dict] | None = None) -> list[dict]:
    """Amendment chains touching `doc_ids`, oldest first, with the quote each document contributes."""
    docs = store.docs
    quotes: dict[str, dict] = {}
    for c in citations or []:
        quotes.setdefault(c["doc_id"], c)
    views = []
    for comp in components(docs):
        if len(comp) < 2 or not (set(comp) & set(doc_ids)):
            continue
        items = []
        for did in sorted(comp, key=lambda i: (eff_date(docs[i]), docs[i]["name"])):
            d = docs[did]
            q = quotes.get(did)
            items.append({"doc_id": did, "name": d["name"], "doc_date": d.get("doc_date"),
                          "effective_date": eff_date(d), "status": status_of(d, as_of),
                          "amends": [docs[t]["name"] for t in d.get("amends", []) if t in docs],
                          "quote": q["quote"] if q else None, "page": q["page"] if q else None})
        views.append({"items": items})
    return views


# ------------------------------------------------------------------ amendment linking
LINK_SCHEMA = {"type": "object", "properties": {"amends": {"type": "array", "items": {"type": "string"}},
                                                "reason": {"type": "string"}}, "required": ["amends", "reason"]}


def _heuristic_links(store, d: dict, others: list[dict]) -> list[str]:
    text = " ".join(c["text"] for c in store.doc_chunks(d["id"])[:3])
    if not _AMEND_WORDS.search(text):
        return []
    out = []
    for o in others:
        stem = re.sub(r"[_\-]+", " ", re.sub(r"\.pdf$", "", o["name"], flags=re.I)).lower()
        words = [w for w in re.findall(r"[a-z]{4,}", stem) if w not in {"amendment", "agreement", "circular"}]
        if words and sum(w in text.lower() for w in words) >= max(1, len(words) // 2) and (o.get("doc_date") or "") <= (d.get("doc_date") or FAR):
            out.append(o["id"])
    return out


def _has_amending_language(store, doc_id: str) -> bool:
    return any(_AMEND_WORDS.search(c["text"]) for c in store.doc_chunks(doc_id))


def link_amendments(store) -> int:
    """Work out which documents amend which. Returns the number of model calls made.

    Incremental: a document is only (re)checked against documents it has not been checked against yet, and
    documents with no amending language at all (a brochure, a base agreement) are skipped without a model call.
    Manual overrides (amends_locked) are always kept."""
    docs = list(store.docs.values())
    calls = 0
    for d in docs:
        others = [o for o in docs if o["id"] != d["id"]]
        checked = set(d.get("link_checked", []))
        fresh = [o for o in others if o["id"] not in checked]
        d["link_checked"] = [o["id"] for o in others]
        if d.get("amends_locked") or not fresh:
            continue
        if not _has_amending_language(store, d["id"]):
            log.info("link: %s has no amending language, skipped", d["name"])
            continue
        snippets = "\n".join(c["text"][:500] for c in store.doc_chunks(d["id"]) if _AMEND_WORDS.search(c["text"]))[:1800]
        head = "\n".join(c["text"] for c in store.doc_chunks(d["id"])[:2])[:1500]
        listing = "\n".join(f'- id={o["id"]} name="{o["name"]}" date={o.get("doc_date")}' for o in fresh)
        system = ("You decide whether a document formally AMENDS, MODIFIES, SUPERSEDES or REPLACES another document. "
                  "A document that merely mentions, references or advertises another document does NOT amend it. "
                  'Reply JSON {"amends": [ids], "reason": str}, using only ids from the list; [] if none.')
        user = f'DOCUMENT "{d["name"]}" (date {d.get("doc_date")})\nOPENING TEXT:\n{head}\n\nAMENDING LANGUAGE FOUND:\n{snippets or "(none)"}\n\nOTHER DOCUMENTS:\n{listing}'
        try:
            calls += 1
            res = llm.chat_json(system, user, LINK_SCHEMA, label=f"link {d['name']}")
            valid = {o["id"] for o in fresh}
            found = [x for x in res.get("amends", []) if x in valid]
            log.info("link: %s amends %s (%s)", d["name"], [store.docs[x]["name"] for x in found], res.get("reason", "")[:80])
        except llm.LLMError as e:
            found = _heuristic_links(store, d, fresh)
            log.warning("link: model failed for %s (%s); heuristic found %d", d["name"], e, len(found))
        d["amends"] = sorted(set(d.get("amends", [])) | set(found))
    store.save()
    return calls


# ------------------------------------------------------------------ conflicts
CONFLICT_SCHEMA = {"type": "object", "properties": {"conflicts": {"type": "array", "items": {
    "type": "object", "properties": {
        "subject": {"type": "string"}, "reason": {"type": "string"},
        "a": {"type": "object", "properties": {"chunk": {"type": "string"}, "quote": {"type": "string"}}, "required": ["chunk", "quote"]},
        "b": {"type": "object", "properties": {"chunk": {"type": "string"}, "quote": {"type": "string"}}, "required": ["chunk", "quote"]}},
    "required": ["subject", "reason", "a", "b"]}}}, "required": ["conflicts"]}

CONFLICT_SYSTEM = (
    "You are a meticulous contract and policy auditor. Given numbered passages, find pairs of statements that "
    "CONFLICT: the SAME subject under the SAME conditions has different numbers, dates, deadlines, percentages, "
    "rules or eligibility conditions. "
    "Do NOT report: statements that agree; differences explained by different conditions (loan slabs, products, "
    "customer types); statements about different subjects; a passage that only describes an amendment without "
    "stating a different value. Quotes must be copied EXACTLY from the passage and contain the conflicting value. "
    'Reply JSON {"conflicts": [{"subject","reason","a":{"chunk","quote"},"b":{"chunk","quote"}}]}; {"conflicts": []} if none.')


def find_conflicts(store, chunks: list[dict], as_of: str) -> list[dict]:
    if len(chunks) < 1:
        return []
    keyed = {f"P{i + 1}": c for i, c in enumerate(chunks)}
    body = "\n\n".join(f'[{k}] ({c["doc"]}, page {c["page"]})\n{c["text"]}' for k, c in keyed.items())
    res = llm.chat_json(CONFLICT_SYSTEM, body, CONFLICT_SCHEMA)
    out, seen = [], set()
    for item in res.get("conflicts", []):
        try:
            ca, cb = keyed.get(item["a"]["chunk"]), keyed.get(item["b"]["chunk"])
            if not ca or not cb:
                continue
            sa, qa, _ = find_quote(item["a"]["quote"], ca["text"], config.QUOTE_MIN_RATIO)
            sb, qb, _ = find_quote(item["b"]["quote"], cb["text"], config.QUOTE_MIN_RATIO)
        except (KeyError, TypeError):
            continue
        if not qa or not qb or qa == qb:                         # unverifiable or identical -> discard
            continue
        na, nb = numbers(qa), numbers(qb)
        if na and nb and na == nb:                               # same numbers: not a numeric conflict
            continue
        key = (frozenset([ca["id"], cb["id"]]), item.get("subject", "").lower().strip())
        if key in seen:
            continue
        seen.add(key)
        verdict = classify(store.docs, ca["doc_id"], cb["doc_id"], as_of)
        out.append({"subject": item.get("subject", ""), "reason": item.get("reason", ""),
                    "a": _side(store, ca, qa), "b": _side(store, cb, qb), **verdict})
    return out


def _side(store, chunk: dict, quote: str) -> dict:
    d = store.docs[chunk["doc_id"]]
    return {"doc_id": chunk["doc_id"], "doc": chunk["doc"], "page": chunk["page"], "chunk": chunk["id"], "quote": quote,
            "doc_date": d.get("doc_date"), "effective_date": eff_date(d)}


def candidate_groups(store, per: int = 3, thr: float | None = None, cap: int | None = None) -> list[list[int]]:
    thr = config.SCAN_SIM if thr is None else thr
    cap = config.SCAN_MAX_GROUPS if cap is None else cap
    has_num = [bool(re.search(r"\d", c["text"])) for c in store.chunks]
    groups, seen = [], set()
    for i, c in enumerate(store.chunks):
        if not has_num[i]:
            continue
        sims = store.emb @ store.emb[i]
        grp = [i]
        for j in np.argsort(-sims):
            j = int(j)
            if j == i:
                continue
            if sims[j] < thr or len(grp) > per:
                break
            if has_num[j]:
                grp.append(j)
        if len(grp) > 1 and frozenset(grp) not in seen:
            seen.add(frozenset(grp))
            groups.append(grp)
    return groups[:cap]


def scan_all(store, as_of: str | None = None, progress=None, lock=None, **kw) -> tuple[list[dict], list[str]]:
    """Scan similar passages across documents for conflicts. Returns (conflicts, errors).

    `lock` is taken per group, so a user's question can run between two groups instead of waiting for the whole scan."""
    as_of = as_of or date.today().isoformat()
    groups = candidate_groups(store, **kw)
    log.info("scan: %d candidate groups", len(groups))
    found, errors, seen = [], [], set()
    for k, grp in enumerate(groups, 1):
        try:
            with (lock or contextlib.nullcontext()):
                items = find_conflicts(store, [store.chunks[i] for i in grp], as_of)
            for c in items:
                key = (c["a"]["chunk"], c["b"]["chunk"])
                if key not in seen and key[::-1] not in seen:
                    seen.add(key)
                    found.append(c)
        except llm.LLMError as e:
            log.warning("scan: group %d/%d failed: %s", k, len(groups), e)
            errors.append(f"group {k}: {e}")
            if isinstance(e, llm.ModelMissing):      # no point continuing
                break
        if progress:
            progress(k, len(groups))
    log.info("scan: %d conflicts, %d errors", len(found), len(errors))
    return found, errors
