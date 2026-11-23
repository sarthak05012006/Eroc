"""Question answering with the zero-guessing pipeline.

retrieve -> draft claims with quotes -> VERIFY every claim in code + a strict judge -> block what fails
-> compute faithfulness -> check conflicts / amendment chain -> answer or abstain."""
from __future__ import annotations

import ast
import copy
import json
import operator
import time
from datetime import date

from . import config, llm, logs, reasoning
from .cache import LRU
from .textutil import canon_num, find_quote, numbers, unsupported_numbers

log = logs.get("qa")
_answers = LRU(config.CACHE_SIZE)


def clear_cache() -> None:
    _answers.clear()

ANSWER_SCHEMA = {"type": "object", "properties": {
    "claims": {"type": "array", "items": {"type": "object", "properties": {
        "text": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "chunk": {"type": "string"}, "quote": {"type": "string"}}, "required": ["chunk", "quote"]}}},
        "required": ["text", "evidence"]}},
    "missing": {"type": "array", "items": {"type": "string"}},
    "calculation": {"type": ["object", "null"], "properties": {
        "description": {"type": "string"}, "expression": {"type": "string"},
        "variables": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "value": {"type": "number"},
            "chunk": {"type": "string"}, "quote": {"type": "string"}}, "required": ["name", "value", "chunk", "quote"]}}}}},
    "required": ["claims", "missing"]}

ANSWER_SYSTEM = """You are a verifiable document assistant for banking, insurance and public-policy PDFs.
RULES
1. Use ONLY the numbered EVIDENCE passages. Never use outside knowledge. Never guess.
2. Write the answer as short claims, one sentence each. Every claim needs 1-3 evidence items {"chunk": "C3", "quote": "..."}.
   Each quote must be COPIED EXACTLY, character for character, from that passage (6-40 words) and must contain the fact.
3. DOCUMENT STATUS tells you how documents relate. An amendment that is in force overrides the text it amends.
   A document marked "not yet in force" does not apply today - mention it as a separate claim with its own evidence.
   If documents that are NOT linked by an amendment disagree, report both statements; do not pick one.
4. If the evidence does not contain what is needed, do not fill the gap: put what is missing in "missing".
5. Reply in the language of the question. Quotes stay in the document's own language.
6. If the question needs arithmetic, fill "calculation": a Python-style expression over named variables, each variable with
   its exact value, chunk and quote. Do NOT compute the result yourself. Otherwise "calculation": null.
Return JSON: {"claims":[{"text","evidence":[{"chunk","quote"}]}],"missing":[...],"calculation":null|{...}}"""

JUDGE_SCHEMA = {"type": "object", "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "integer"}, "verdict": {"type": "string", "enum": ["supported", "partial", "unsupported", "contradicted"]},
    "reason": {"type": "string"}}, "required": ["id", "verdict", "reason"]}}}, "required": ["results"]}

JUDGE_SYSTEM = ("You are a strict fact-checker. For each item decide whether the QUOTES alone fully support the CLAIM. "
                "Numbers, dates, names, conditions and scope in the claim must all match the quotes. "
                '"supported" = fully; "partial" = only part; "unsupported" = quotes do not say it; "contradicted" = quotes say otherwise. '
                'Reply JSON {"results":[{"id","verdict","reason"}]}.')


# ------------------------------------------------------------------ helpers
def build_evidence(store, hits: list[dict], qvec=None) -> list[dict]:
    """Evidence for the model, in priority order:
    1. the top search hits,
    2. the best passage of EVERY document linked to them by an amendment (so each version of a rule is seen),
    3. neighbouring chunks of the best hits (answers that continue over a page break)."""
    ordered, seen = [], set()

    def add(c):
        if c and c["id"] not in seen and len(ordered) < config.MAX_EVIDENCE:
            seen.add(c["id"])
            ordered.append(c)

    for h in hits:
        add(h["chunk"])
    if qvec is not None and store.docs:
        top_docs = {h["chunk"]["doc_id"] for h in hits[:4]}
        related: set[str] = set()
        for comp in reasoning.components(store.docs):
            if set(comp) & top_docs and len(comp) > 1:
                related |= set(comp)
        have = {c["doc_id"] for c in ordered}
        best = store.best_chunks_for_docs(qvec, [d for d in related if d not in have])
        for chunk, sim in sorted(best.values(), key=lambda x: -x[1]):
            if sim >= config.MIN_SIM:
                add(chunk)
    for h in hits[:3]:
        c = h["chunk"]
        add(store.neighbor(c, 1))
        if c["text"][:1].islower():
            add(store.neighbor(c, -1))
    return [{"key": f"C{i + 1}", "chunk": c} for i, c in enumerate(ordered)]


def status_lines(store, as_of: str) -> str:
    lines = []
    for d in sorted(store.docs.values(), key=lambda x: reasoning.eff_date(x)):
        amends = ", ".join(store.docs[t]["name"] for t in d.get("amends", []) if t in store.docs)
        lines.append(f'- {d["name"]}: dated {d.get("doc_date")}, effective {reasoning.eff_date(d)}, '
                     f'{reasoning.status_of(d, as_of).upper()}' + (f", amends: {amends}" if amends else ""))
    return "\n".join(lines)


def _cite(item: dict, snapped: str, score: float) -> dict:
    c = item["chunk"]
    return {"doc_id": c["doc_id"], "doc": c["doc"], "page": c["page"], "chunk": c["id"], "quote": snapped,
            "match": round(score, 3), "kind": c["kind"]}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod}
_FUNCS = {"min": min, "max": max, "round": round, "abs": abs}


def safe_eval(expr: str, names: dict[str, float]) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.Name) and n.id in names:
            return names[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 1000:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr.strip(), mode="eval"))


def run_calculation(calc: dict | None, evid: dict[str, dict]) -> dict | None:
    """Every input must be quoted from the documents; the arithmetic is done by Python, not the LLM."""
    if not calc or not calc.get("expression"):
        return None
    names, steps = {}, []
    for v in calc.get("variables", []):
        item = evid.get(v.get("chunk"))
        if not item:
            return {"ok": False, "error": f'input "{v.get("name")}" cites a passage that does not exist'}
        score, snapped, _ = find_quote(v.get("quote", ""), item["chunk"]["text"], config.QUOTE_MIN_RATIO)
        if not snapped or canon_num(str(v["value"])) not in numbers(snapped):
            return {"ok": False, "error": f'input "{v.get("name")}" = {v.get("value")} is not in its quoted evidence'}
        names[v["name"]] = v["value"]
        steps.append({"name": v["name"], "value": v["value"], "cite": _cite(item, snapped, score)})
    try:
        result = safe_eval(calc["expression"], names)
    except Exception as e:
        return {"ok": False, "error": f"could not evaluate: {e}"}
    return {"ok": True, "description": calc.get("description", ""), "expression": calc["expression"],
            "inputs": steps, "result": round(float(result), 2)}


def _result(question, as_of, status, **kw) -> dict:
    base = {"question": question, "as_of": as_of, "status": status, "answer": [], "blocked": [], "missing": [],
            "faithfulness": {"score": 0.0, "supported": 0, "total": 0}, "conflicts": [], "chain": [],
            "calculation": None, "warnings": [], "retrieval": {"best_sim": 0.0, "chunks": []}, "seconds": 0.0,
            "timings": {}, "cached": False}
    base.update(kw)
    return base


# ------------------------------------------------------------------ main entry
def _stage(cb, name: str, **kw):
    if cb:
        cb(name, **kw)


def ask(store, question: str, as_of: str | None = None, doc_ids: list[str] | None = None,
        check_conflicts: bool = True, on_stage=None) -> dict:
    """Answer `question` from the stored documents. `on_stage(name, **info)` is called as the pipeline moves on
    (retrieving, drafting, verifying, conflicts) so a UI can show progress."""
    t0 = time.time()
    as_of = as_of or date.today().isoformat()
    date.fromisoformat(as_of)                                   # ValueError on a bad date; the API turns it into a 400
    key = (question.strip().lower(), as_of, tuple(sorted(doc_ids or [])), check_conflicts, store.version, config.LLM_MODEL)
    hit = _answers.get(key)
    if hit is not None:
        out = copy.deepcopy(hit)
        out.update(cached=True, seconds=round(time.time() - t0, 2))
        log.info("ASK (cached) status=%s", out["status"])
        return out

    if not store.docs:
        return _result(question, as_of, "abstained", missing=["No documents have been uploaded yet."])

    timings: dict[str, float] = {}
    tick = time.time()

    def lap(name):
        nonlocal tick
        now = time.time()
        timings[name] = round(now - tick, 1)
        tick = now

    _stage(on_stage, "retrieving")
    qv = llm.embed_query(question)
    hits = store.search(qv, question, config.TOP_K, doc_ids)
    best = max((h["sim"] for h in hits), default=0.0)
    lap("retrieve")
    if not hits or best < config.MIN_SIM:
        log.info("ASK abstained before the model: best_sim=%.3f < MIN_SIM=%.2f", best, config.MIN_SIM)
        r = _result(question, as_of, "abstained", missing=[
            "No passage in the uploaded documents is relevant to this question, so I have no proof to answer from."],
            retrieval={"best_sim": round(best, 3), "chunks": []}, timings=timings)
        r["seconds"] = round(time.time() - t0, 1)
        return r

    evidence = build_evidence(store, hits, qv)
    evid = {e["key"]: e for e in evidence}
    ctx = "\n\n".join(f'[{e["key"]}] {e["chunk"]["doc"]}, page {e["chunk"]["page"]}'
                      + (f', section "{e["chunk"]["section"]}"' if e["chunk"]["section"] else "")
                      + f'\n{e["chunk"]["text"]}' for e in evidence)
    user = (f"AS-OF DATE: {as_of}\nDOCUMENT STATUS:\n{status_lines(store, as_of)}\n\n"
            f"EVIDENCE:\n{ctx}\n\nQUESTION: {question}")
    warnings: list[str] = []

    _stage(on_stage, "drafting", evidence=len(evidence))
    draft = llm.chat_json(ANSWER_SYSTEM, user, ANSWER_SCHEMA, label="answer")
    claims = [c for c in draft.get("claims", []) if isinstance(c, dict) and str(c.get("text", "")).strip()]
    missing = [str(m) for m in draft.get("missing", []) if str(m).strip()]
    lap("draft")

    # ---- 1. deterministic checks: quote must exist verbatim; numbers must be in the quote
    _stage(on_stage, "verifying", claims=len(claims))
    blocked, pending = [], []
    for c in claims:
        cites, why = [], []
        for ev in (c.get("evidence") or [])[:3]:
            item = evid.get(ev.get("chunk"))
            if not item:
                why.append(f'it cites passage {ev.get("chunk")}, which was not retrieved')
                continue
            score, snapped, _ = find_quote(ev.get("quote", ""), item["chunk"]["text"], config.QUOTE_MIN_RATIO)
            if not snapped:
                why.append("its quote does not appear verbatim in the cited document")
                continue
            cites.append(_cite(item, snapped, score))
        if not cites:
            blocked.append({"text": c["text"], "reason": "Blocked: " + (why[0] if why else "no evidence was given for it")})
            continue
        bad = unsupported_numbers(c["text"], [x["quote"] for x in cites], [question, as_of])
        if bad:
            blocked.append({"text": c["text"], "reason": f'Blocked: the figure(s) {", ".join(bad)} are not in the quoted evidence'})
            continue
        pending.append((c, cites))

    # ---- 2. strict judge on what survived (fails closed if the judge is unavailable)
    verified: list[dict] = []
    if pending:
        payload = [{"id": i, "claim": c["text"], "quotes": [x["quote"] for x in cites]} for i, (c, cites) in enumerate(pending)]
        verdicts = {}
        try:
            res = llm.chat_json(JUDGE_SYSTEM, json.dumps(payload, ensure_ascii=False), JUDGE_SCHEMA, label="judge")
            verdicts = {r["id"]: r for r in res.get("results", []) if isinstance(r, dict) and "id" in r}
        except llm.LLMError as e:
            log.error("judge unavailable: %s", e)
            warnings.append("The verifier model was unreachable, so unverified claims were blocked.")
        for i, (c, cites) in enumerate(pending):
            v = verdicts.get(i)
            if v and v.get("verdict") == "supported":
                verified.append({"text": c["text"], "citations": cites})
            else:
                why = (v or {}).get("verdict", "unverified")
                blocked.append({"text": c["text"], "reason": f'Blocked by verifier ({why}): {(v or {}).get("reason", "no verdict returned")}'})
    lap("verify")

    total = len(claims)
    faith = {"supported": len(verified), "total": total, "score": round(100 * len(verified) / total, 1) if total else 0.0}

    # ---- 3. calculation (inputs quoted, arithmetic in Python)
    calc = run_calculation(draft.get("calculation"), evid) if verified or draft.get("calculation") else None

    # ---- 4. contradictions + amendment chain over the retrieved evidence
    conflicts: list[dict] = []
    if check_conflicts and len(evidence) >= 2:
        _stage(on_stage, "conflicts")
        try:
            conflicts = reasoning.find_conflicts(store, [e["chunk"] for e in evidence][:8], as_of)
        except llm.LLMError as e:
            log.error("contradiction check failed: %s", e)
            warnings.append("Contradiction check could not run.")
        lap("conflicts")
    cited = [c for v in verified for c in v["citations"]]
    doc_ids_touched = {c["doc_id"] for c in cited} | {s["doc_id"] for k in conflicts for s in (k["a"], k["b"])}
    chain = reasoning.chain_view(store, list(doc_ids_touched), as_of, cited)

    # ---- 5. final status
    unresolved = [k for k in conflicts if k["status"] == "contradiction"]
    if not verified and not calc:
        status = "abstained"
        if not missing:
            missing = ["The documents do not contain a statement I could verify against this question."]
    elif unresolved:
        status = "conflict"
    elif blocked or missing:
        status = "partial"
    else:
        status = "verified"

    result = _result(question, as_of, status, answer=verified, blocked=blocked, missing=missing, faithfulness=faith,
                     conflicts=conflicts, chain=chain, calculation=calc, warnings=warnings, timings=timings,
                     retrieval={"best_sim": round(best, 3), "chunks": [e["chunk"]["id"] for e in evidence]},
                     seconds=round(time.time() - t0, 1))
    log.info("ASK status=%s faithfulness=%s%% (%d/%d) conflicts=%d blocked=%d best_sim=%.3f %.1fs %s",
             status, faith["score"], faith["supported"], faith["total"], len(conflicts), len(blocked), best,
             result["seconds"], timings)
    for b in blocked:
        log.info("  blocked: %s", b["reason"])
    for m in missing:
        log.info("  missing: %s", m)
    if not warnings:                                            # never cache a degraded answer
        _answers.set(key, copy.deepcopy(result))
    return result
