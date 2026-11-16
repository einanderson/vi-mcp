"""Collect mode of vi_recall (0.67.0): a question that wants many values about one entity ("what hardware does host X have") gets the
relevant chunks of every object about it, chosen under a token budget — not four hits.

The user decided (16.11.2026) that the memory recognises such questions itself. Measured on a production copy: 20 % of all objects are
parts of longer pages ("(part 2/3)", 3,387 pages, up to 159 parts), and the facts about one host are spread over several pages; top-k
returns about a quarter of a many-answer set even at k = 20 (QUEST 2305.11694, QAMPARI 2205.12665).

Steps (research VI erkenntnis:8200b28e…):
1. detect: the question names an entity that exists as a hashtag value (host:…) and either asks to collect ("hardware", "all",
   "which", "overview", …) or consists of little more than the entity — every decision is returned with its reason, so router accuracy
   can be measured;
2. candidates: the search hits ∪ every visible object carrying the entity hashtag (an exhaustive filter, not a ranking: single-vector
   retrieval cannot return complete sets, LIMIT 2508.21038; dense retrieval is weak on rare names, EntityQuestions 2109.08535);
3. group by page (data.source without its version suffix), parts in their order, chunked by `chunking`;
4. choose with `context_budget` (reranker per chunk, neighbours in the same section, a cap per page, chunks naming the entity first, cut
   at the largest score drop between a minimum and a maximum token budget), output in page order with […] for gaps;
5. name what was left out: per page the parts kept of all, and the counts with their reason.
The hashtag is noisy (pages that only mention a host carry it too), so nothing is taken unscored."""
from __future__ import annotations

import re
from typing import Any

from . import chunking, context_budget as cb, memory as g, vector

#: hashtag kinds whose values are entities a question can be about
ENTITY_KINDS = ("host",)
#: words that ask for many values about the entity (German and English; the memory's users write both)
INTENT = re.compile(r"\b(ausstattung|hardware|alle[snm]?|alles|welche[srnm]?|übersicht|ueberblick|überblick|liste|konfiguration|"
                    r"eingerichtet|details?|inventar|was\s+weißt|was\s+weisst|wie\s+ist\s+.*\s+aufgebaut|all|which|overview|list|"
                    r"profile|configuration|setup|inventory|everything|what\s+do\s+you\s+know)\b", re.I)
#: tokens that look like entity names: letters and digits mixed, optionally with a domain (host7, host7.example.org)
TOKEN = re.compile(r"\b([a-z][a-z0-9_-]*\d[a-z0-9_-]*(?:\.[a-z0-9-]+)*)\b", re.I)
STOP = {"der", "die", "das", "den", "dem", "des", "ein", "eine", "hat", "ist", "von", "zu", "auf", "im", "in", "und", "the", "a", "an",
        "of", "on", "for", "is", "has", "what", "was", "wie", "how", "about", "über", "ueber"}
PART = re.compile(r"\((?:Teil|part|Text-Teil|Anhang-Teil)\s+(\d+)\s*/\s*(\d+)\)", re.I)

MIN_TOKENS, MAX_TOKENS, PAGE_CORE, MAX_PIECES = 1500, 8000, 3, 240


def entity_tags(text: str) -> list[str]:
    """Hashtags (kind:value) of ENTITY_KINDS whose value starts with a host-like token of the text, e.g. host7 → host:host7.example.org."""
    found: list[str] = []
    for tok in dict.fromkeys(t.lower() for t in TOKEN.findall(text or "")):
        for kind in ENTITY_KINDS:
            with g.lock:
                rows = g.db().execute("SELECT DISTINCT tag FROM tag WHERE tag = ? OR tag LIKE ? LIMIT 5",
                                      (f"{kind}:{tok}", f"{kind}:{tok}.%")).fetchall()
            found += [r[0] for r in rows if r[0] not in found]
    return found


def detect(filter_text: str) -> dict[str, Any] | None:
    """The router: None for a single-fact search, else {entities, reason}."""
    if not filter_text or ":" in filter_text.split()[0] and not TOKEN.search(filter_text):
        return None
    ents = entity_tags(filter_text)
    if not ents:
        return None
    intent = INTENT.search(filter_text)
    rest = [w for w in re.findall(r"[\wäöüß.-]+", filter_text.lower()) if w not in STOP and not TOKEN.fullmatch(w)]
    if intent:
        return {"entities": ents, "reason": f"entity {', '.join(ents)} with collect intent “{intent.group(0)}”"}
    if len(rest) <= 1:
        return {"entities": ents, "reason": f"the question is the entity {', '.join(ents)} itself"}
    return None


def _page_key(o) -> str:
    src = str(o.data.get("source") or "").split(" · ")[0].strip()
    return src or o.id


def _part(o) -> tuple[int, int]:
    m = PART.search(o.text or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _title(o) -> str:
    t = PART.sub("", (o.text or "").split(" – ")[0]).strip(" :")
    return t[:120]


def collect(query_words: str, hits: list, decision: dict[str, Any], *, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS,
            per_page: int = PAGE_CORE) -> tuple[str, dict[str, Any]]:
    """The collected context for the question and its counts. `hits`: the search's (score, object, weights)."""
    counts: dict[str, Any] = {"entities": decision["entities"], "reason": decision["reason"]}
    entity_ids: list[str] = []
    for tag in decision["entities"]:
        entity_ids += [i for i in g.ids_for_filter(tag) if i not in entity_ids]
    counts["objects_with_entity_tag"] = len(entity_ids)
    order = [o.id for _s, o, _w in hits] + [i for i in entity_ids if i not in {o.id for _s, o, _w in hits}]
    objs = g.load_many(order)
    pages: dict[str, list] = {}
    for oid in order:  # pages in the order of their best object: search rank first, then the entity's other objects
        if oid in objs:
            pages.setdefault(_page_key(objs[oid]), []).append(objs[oid])
    counts["pages"] = len(pages)
    pieces: list[cb.Piece] = []
    texts: dict[int, list[str]] = {}
    meta: dict[int, dict[str, Any]] = {}
    for entry, (key, parts) in enumerate(pages.items()):
        parts.sort(key=lambda o: _part(o)[0])
        title = _title(parts[0])
        own: list[str] = []
        for o in parts:
            content = str(o.data.get("content") or "").strip() or o.text
            for c in chunking.chunk_markdown(title, content):
                own.append(c.text)
        texts[entry] = own
        n_parts = max((_part(o)[1] for o in parts), default=0)
        meta[entry] = {"title": title, "source": key, "ids": [o.id for o in parts], "parts_here": len(parts), "parts_total": n_parts}
        for pos, t in enumerate(own):
            pieces.append(cb.Piece(entry, pos, t, cb.section_of(t)))
    counts["chunks"] = len(pieces)
    if len(pieces) > MAX_PIECES:  # pages are in rank order, so the tail is the least relevant; counted, never silent
        counts["chunks_not_scored"] = len(pieces) - MAX_PIECES
        pieces = pieces[:MAX_PIECES]
    if not pieces:
        return "", counts
    try:
        scores = vector.rerank_scores(query_words, [p.text for p in pieces]) if vector.rerank_available() else None
        counts["scored_by"] = "reranker" if scores is not None else "word overlap"
    except Exception as ex:  # noqa: BLE001 — the collect mode must never cost the answer: word overlap, named
        scores, counts["scored_by"] = None, f"word overlap (reranker failed: {type(ex).__name__})"
    if scores is None:
        q = set(re.findall(r"\w+", query_words.lower())) - STOP
        scores = [len(q & set(re.findall(r"\w+", p.text.lower()))) for p in pieces]
    names = {t.split(":", 1)[1].split(".")[0] for t in decision["entities"]}
    forced = {i for i, p in enumerate(pieces) if any(n in p.text.lower() for n in names)}
    header_tokens = {e: cb.estimate_tokens(f"{m['title']} {m['source']}") + 8 for e, m in meta.items()}
    sel = cb.select(pieces, scores, header_tokens, min_tokens=min_tokens, max_tokens=max_tokens, per_entry_core=per_page, forced=forced)
    counts.update(tokens=sel.tokens, dropped=sel.dropped, pages_kept=len(sel.keep))
    out: list[str] = []
    for n, entry in enumerate(sorted(sel.keep), 1):
        m = meta[entry]
        kept = sel.keep[entry]
        parts = f"parts {m['parts_here']} of {m['parts_total']} found" if m["parts_total"] else "one object"
        out.append(f"[{n}] {m['title']} · {m['source']} · {', '.join(m['ids'][:4])}{' …' if len(m['ids']) > 4 else ''} · "
                   f"{len(kept)} of {len(texts[entry])} chunks · {parts}\n{cb.render(texts[entry], kept)}")
    return "\n\n".join(out), counts


def note(counts: dict[str, Any]) -> str:
    """One line on what the collect mode did and left out (nothing disappears silently)."""
    d = counts.get("dropped") or {}
    left = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in d.items() if v)
    extra = f", {counts['chunks_not_scored']} chunks of the last pages not scored" if counts.get("chunks_not_scored") else ""
    return (f"collect mode ({counts['reason']}): {counts.get('pages_kept', 0)} of {counts['pages']} pages, "
            f"{counts.get('tokens', 0)} tokens, {counts['objects_with_entity_tag']} objects carry the entity hashtag, chunks scored by "
            f"{counts.get('scored_by', '–')}" + (f"; left out: {left}" if left else "") + extra)
