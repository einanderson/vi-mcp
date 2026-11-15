"""Provenance adapter: the evidence about a knowledge object as input facts of kollektiv 1.3.0 — shadow mode only (Unreleased).

kollektiv 1.3.0 (P8–P12) asks where a statement's support comes from: copies (derives_from), model-made origins (machine_made), the
weight of each origin (evidence_kind), camps (camp), testability and extraordinary claims. This module turns what the memory STORES
about one object into those facts. It never guesses: an input the data does not hold stays absent, and the seed treats an absent input
as "not known", never as evidence. It lives in vi_mcp (clingo_object never imports vi_mcp) and writes nothing: `vi-mcp philosophy
shadow kollektiv` compares the seed's acceptance with today's object mode and counts the disagreements.

The mapping, input by input (memory.Objekt.confirm/refute, vil.evidence_of):
- author(S, A)        the object's instance.
- line(S, P, O, I)    one per stored confirmation (pos) and refutation (neg) entry: I is the instance that gave it, O its origin — the
                      cited source when there is one (one source cited by many confirmers is ONE origin, so copies of a citation add
                      nothing), an object it names (obj:<id>), else the confirming instance's own check (i:<instance>). An object
                      from before the entries (0.33.0) has only instance counts (bestaetigt_von / widerlegt_von): a line from
                      i:<instance> each.
- closed(S)           the record is complete: the object's counters (bestaetigt, widerlegt) equal the entries kept. confirm/refute keep
                      the last 100 entries but count every call, so a longer record lost its oldest entries — their origins are
                      unknown, so they are neither invented (no line) nor ignored (not closed: nothing is concluded from absence).
                      Never closed in quarantine.
- camp(O, C)          the model family stored with an entry, for every origin that has one. A partly known camp is the seed's to
                      judge (P10 holds back with camp_unknown, never one_camp): one place decides.
- withdrawn(S)        the author's own refutation entry exists (or, before the entries, the author's refutation count): the author
                      withdrew the claim. P13: that removes the claim and proves nothing against it; the author's line still counts
                      in neither direction (P1). Since 0.67.0 memory names it too: mode `zurueckgezogen` (withdrawn) apart from
                      `widerlegt` (refuted by two foreign families); an unmigrated `widerlegt` is still told apart here.
- derives_from(O, R)  an origin that is a memory object whose own source names another object: obj:<id> derives from obj:<that id>,
                      followed along the chain.
- machine_made(O)     an origin a model produced with no source: the server's own instance (automatic, i:vi) without a cited source, or
                      an object written by the sleep's language model (origin:sleep-llm) whose source is no URL.
- evidence_kind(R, K) a cited URL → document. An origin that is an object → vil.evidence_of mapped: documented → document,
                      hearsay → hearsay, inferred → hearsay, measured → none, unknown → none.
                      inferred → hearsay, not document: an inferred object is a language model's conclusion from other material —
                      nobody observed or wrote it down as a source; calling it a document would give it the weight of a record.
                      measured → none: vil calls an object "measured" when following it as a HINT was counted (bandit outcomes) — that
                      measures usefulness as a hint, not the statement; one word, two meanings, so it is not carried over.
- source(S, pos, R, web)  the object's own URL source (P7). There is no verification record in the memory, so no verified(R, S):
                      such a source counts nothing and gives the duty to verify it.
- foreign(S)          imported from a network (instance p2p-…, origin:p2p); unreviewed(S): in quarantine.
- untestable, testable_by, extraordinary: never filled — the memory holds no such judgement.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from . import memory as g
from .code_facts import _q  # one quoting of a string term (the architecture rule no_duplicates found the copy)

KINDS = ("insight", "rule", "skill")
OBJ_REF = re.compile(r"\b((?:erkenntnis|regel|skill|erfahrung|insight|rule|experience):[0-9A-Za-z_-]{6,})\b")  # ids of 0.66 and later
URL = re.compile(r"https?://\S+")
VIL_TO_KIND = {"documented": "document", "hearsay": "hearsay", "inferred": "hearsay"}  # measured, unknown → no kind (docstring)


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())[:200]


def _origin(e: dict) -> str:
    src = str(e.get("quelle") or "").strip()
    if not src:
        return f"i:{e.get('instanz')}"
    ref = OBJ_REF.search(src)
    if ref and not URL.search(src):
        return f"obj:{ref.group(1)}"
    return f"q:{_norm(src)}"


def facts_for(o: g.MemoryObject, load=g.load) -> tuple[list[str], dict]:
    """(facts in clingo-object notation, what was filled and why) for one knowledge object."""
    s = _q(o.id)
    d = o.data
    out = [f"author({s}, {_q(o.instance)})"]
    info: Counter = Counter()
    origins: dict[str, dict] = {}  # origin -> {"camp": family or None}
    complete = True
    for key, legacy, pol, counter in (("bestaetigungen", "bestaetigt_von", "pos", o.confirmed), ("widerlegungen", "widerlegt_von", "neg", o.refuted)):
        entries = list(d.get(key) or [])
        counts = dict(d.get(legacy) or {})
        for e in entries:
            org = _origin(e)
            out.append(f"line({s}, {pol}, {_q(org)}, {_q(e.get('instanz'))})")
            origins.setdefault(org, {"camp": e.get("familie"), "entry": e})
            info[f"line_{pol}"] += 1
        if not entries and counts and sum(counts.values()) == counter:  # an object from before the entries: instance counts only
            for inst in counts:
                org = f"i:{inst}"
                out.append(f"line({s}, {pol}, {_q(org)}, {_q(inst)})")
                origins.setdefault(org, {"camp": None, "entry": {"instanz": inst}})
                info[f"line_{pol}_legacy"] += 1
        elif counter != len(entries):  # entries dropped at the cap, or calls without an entry: origins unknown
            complete = False
            info["not_closed_record"] += 1
    if o.mode == "quarantine":
        complete = False
        info["not_closed_quarantine"] += 1
        out.append(f"unreviewed({s})")
    if complete:
        out.append(f"closed({s})")
    camps = {org: x["camp"] for org, x in origins.items() if x["camp"]}
    out += [f"camp({_q(org)}, {_q(c)})" for org, c in camps.items()]
    info["camps"] += len(camps)
    if o.instance in _refuters(o) or o.mode == "withdrawn":  # 0.67.0: the mode itself, also for a server-withdrawn skill
        out.append(f"withdrawn({s})")
        info["withdrawn"] += 1
    for org, x in origins.items():
        e = x["entry"]
        if org.startswith("q:"):
            if URL.search(str(e.get("quelle") or "")):
                out.append(f"evidence_kind({_q(org)}, document)")
                info["kind_document"] += 1
        elif org.startswith("i:"):
            if e.get("instanz") == g.SERVER_INSTANCE:
                out.append(f"machine_made({_q(org)})")
                info["machine_made"] += 1
        else:
            out += _object_origin(org, load, info, set())
    src = str(d.get("quelle") or "")
    url = URL.search(src)
    if url:
        out.append(f"source({s}, pos, {_q('url:' + url.group(0)[:300])}, web)")
        info["source_unverified"] += 1
    if str(o.instance or "").startswith("p2p-") or "origin:p2p" in o.hashtags:
        out.append(f"foreign({s})")
        info["foreign"] += 1
    return out, dict(info)


def _object_origin(org: str, load, info: Counter, seen: set) -> list[str]:
    """An origin that is a memory object: its kind of evidence, whether a model made it, and whom it copies (followed along)."""
    from . import vil
    if org in seen:
        return []
    seen.add(org)
    o = load(org[4:])
    if o is None:
        info["origin_object_missing"] += 1
        return []
    out = []
    kind = VIL_TO_KIND.get(vil.evidence_of(o))
    if kind:
        out.append(f"evidence_kind({_q(org)}, {kind})")
        info[f"kind_{kind}"] += 1
    src = str(o.data.get("quelle") or "")
    if "origin:sleep-llm" in o.hashtags and not URL.search(src):
        out.append(f"machine_made({_q(org)})")
        info["machine_made"] += 1
    ref = OBJ_REF.search(src)
    if ref and not URL.search(src) and ref.group(1) != o.id:
        up = f"obj:{ref.group(1)}"
        out.append(f"derives_from({_q(org)}, {_q(up)})")
        info["derives_from"] += 1
        out += _object_origin(up, load, info, seen)
    return out


# ── shadow comparison ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _refuters(o: g.MemoryObject) -> set[str]:
    return {e.get("instanz") for e in o.data.get("widerlegungen") or []} | set(o.data.get("widerlegt_von") or {})


def today(o: g.MemoryObject) -> str:
    """Today's verdict class of an object (exhaustive): withdrawn · refuted_by_author · refuted · disputed · confirmed_2plus ·
    confirmed_1 · unconfirmed · mode_<other>. Since 0.67.0 memory sets `zurueckgezogen` for a withdrawal (P13) and `widerlegt` only for
    two foreign families; a `widerlegt` from before the migration is still told apart (refuted_by_author)."""
    if o.mode == "withdrawn":
        return "withdrawn"
    if o.mode == "refuted":
        return "refuted_by_author" if o.instance in _refuters(o) and len(o._lines("widerlegungen", "widerlegt_von")) < 2 else "refuted"
    if o.mode == "disputed":
        return "disputed"
    if o.mode != "active":
        return f"mode_{o.mode}"
    n = o.foreign_confirmations()
    return "confirmed_2plus" if n >= 2 else "confirmed_1" if n == 1 else "unconfirmed"


#: the seed's acceptance that matches today's class; anything else is a disagreement
AGREES = {"withdrawn": {"withdrawn"}, "refuted_by_author": {"withdrawn"}, "refuted": {"rejected"}, "disputed": {"unresolved"}, "confirmed_2plus": {"accepted"}, "confirmed_1": {"tentative"},
          "unconfirmed": {"unresolved"}}


def _solve(pkg: dict, facts: list[str]) -> set[str]:
    from clingo_object import notation as n
    from clingo_object import philosophy as ph
    program, stmts, _f = n.translate_full(ph.program(pkg))
    asp = "\n".join(_asp(f) for f in facts)
    raw = n.run_program(asp + "\n" + program, stmts, n.prior_belief, None)
    models = [set(m) for m in raw.get("models") or []]
    return {"".join(a.split()) for a in (set.intersection(*models) if models else set())}


def _asp(fact: str) -> str:
    """A notation fact as an ASP fact: the facts here are plain atoms of quoted strings and constants, valid ASP once ended by a dot."""
    return fact + "."


def shadow(limit: int | None = None, chunk: int = 2000, examples: int = 5) -> dict:
    """Compare kollektiv's acceptance with today's object mode over the knowledge objects. Writes nothing."""
    from clingo_object import philosophy as ph
    pkg = ph.load("kollektiv")
    with g.lock:
        ids = [r[0] for r in g.db().execute(
            f"SELECT id FROM object WHERE kind IN ({','.join('?' * len(KINDS))}) AND mode NOT IN ('archived', 'merged') ORDER BY id",
            KINDS)]
    ids = ids[:limit] if limit else ids
    pairs: Counter = Counter()
    held: Counter = Counter()
    filled: Counter = Counter()
    ex: dict[str, list[str]] = {}
    for k in range(0, len(ids), chunk):
        objs = [o for o in g.load_many(ids[k:k + chunk]).values()]
        facts: list[str] = []
        for o in objs:
            fs, info = facts_for(o)
            facts += fs
            filled.update(info)
        atoms = _solve(pkg, facts)
        acc = {}
        for a in atoms:
            m = re.fullmatch(r'acceptance\("(.+)",(\w+)\)', a)
            if m:
                acc.setdefault(m.group(1), set()).add(m.group(2))
            m = re.fullmatch(r'grund\("(.+)",held_back,(\w+)\)', a)
            if m:
                held[m.group(2)] += 1
        for o in objs:
            t = today(o)
            seed = "/".join(sorted(acc.get(o.id, set()))) or "none"
            pairs[(t, seed)] += 1
            if t in AGREES and seed not in AGREES[t]:
                ex.setdefault(f"{t} -> {seed}", [])
                if len(ex[f"{t} -> {seed}"]) < examples:
                    ex[f"{t} -> {seed}"].append(o.id)
    agree = sum(v for (t, s), v in pairs.items() if s in AGREES.get(t, set()))
    comparable = sum(v for (t, _s), v in pairs.items() if t in AGREES)
    return {"package": f"{pkg['name']} {pkg['version']}", "hash": ph.package_hash(pkg), "objects": len(ids),
            "comparable": comparable, "agree": agree, "disagree": comparable - agree,
            "pairs": {f"{t} -> {s}": v for (t, s), v in sorted(pairs.items(), key=lambda kv: -kv[1])},
            "held_back": dict(held), "filled": dict(sorted(filled.items())), "examples": ex,
            "never_filled": ["untestable", "testable_by", "extraordinary", "verified"]}


def shadow_lines(r: dict) -> list[str]:
    out = [f"Shadow comparison, {r['package']} (hash {r['hash'][:16]}…) — nothing was changed",
           f"  {r['objects']} knowledge objects, {r['comparable']} comparable: {r['agree']} agree, {r['disagree']} disagree",
           "  today -> seed:"] + [f"    {k}: {v}" for k, v in r["pairs"].items()]
    if r["held_back"]:
        out.append("  held back by the seed: " + ", ".join(f"{k} {v}" for k, v in r["held_back"].items()))
    out.append("  inputs filled: " + (", ".join(f"{k} {v}" for k, v in r["filled"].items()) or "none"))
    out.append("  never filled (no such data in the memory): " + ", ".join(r["never_filled"]))
    for k, ids in r["examples"].items():
        out.append(f"  e.g. {k}: {', '.join(ids)}")
    return out


def main_shadow(name: str, limit: int | None, db: str | None, out_json: str | None) -> int:
    if name != "kollektiv":
        raise SystemExit(f"shadow fills the inputs of kollektiv only (its vocabulary is what this adapter knows), not {name}")
    from . import config
    g.open_db(db or config.load().db)
    try:
        r = shadow(limit)
    finally:
        g.close_db()
    print("\n".join(shadow_lines(r)))
    if out_json:
        from pathlib import Path
        Path(out_json).write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0
