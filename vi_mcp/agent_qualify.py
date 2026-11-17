"""The AI that uses the VI qualifies what it reads (0.69.3).

Why (the user, 17.11.2026): "die KI, die die VI nutzt, soll das übernehmen. Wenn ein Objekt sowieso angefasst wird und im Kontext
steht, sollen Reflexion, Bewertung, Prüfung und Frage-Antwort-Paare übernommen werden." The server's own model writes training pairs
for 5 objects per night (~40 s each on a shared model); an agent that just read an object with vi_object has its whole text in context
already, so a few grounded pairs and two labels cost it ~250–400 output tokens, once per object.

How:
- vi_object ends with a short request when the object (knowledge, not private, not in quarantine) has no current training pairs or no
  current labels — at most once per object and caller per day (REQUEST_COOLDOWN_S), never for code pieces;
- the agent answers with vi_qualify(id, pairs, answers, memory): the pairs pass the same fixed rules as the server's (qa_pairs.select:
  self-contained question, not copied, every name and number of the answer in the document); one kept pair becomes the probe
  (qa_pairs.split_probe); the record names the agent (version.model "agent:<family or instance>") and that no model support check ran;
- labels answers:/memory: from the fixed classes (classify.ANSWERS, classify.MEMORY);
- tags (kind:value, e.g. host:, service:, topic:) in the same call — tagging and qualifying are one step (the user, 17.11.2026); fixed
  kinds (instance:, mode:, …) and malformed tags are refused one by one and named, at most MAX_TAGS;
- review and rating stay where they are: vi_confirm / vi_refute when the agent sees that the object is right or wrong, vi_feedback for
  hints, vi_rate for its own conversation. An agent may review its own work (the user, 17.11.2026: "in our environment there is only
  Claude, so the AI must be able to check its own work"): it counts marked "self" at memory.SELF_WEIGHT; another instance counts 1.
  An agent's verdict never releases anything from quarantine — that stays with a human or the release right.
Agent-written pairs count as current: sleep step 8i does not redo them (AGENT_PROMPT in classify.pairs_pending).
"""
from __future__ import annotations

import time

from . import memory as g

AGENT_PROMPT = "agent-v1"
REQUEST_COOLDOWN_S = 86400
MAX_TAGS = 10
#: kinds the server sets by fixed rules (doc_triggers) — an object without any of them lacks triggers
TRIGGER_KINDS = ("host:", "service:", "topic:", "tool:", "product:")
_asked: dict[tuple[str, str], float] = {}


def _eligible(o: g.MemoryObject) -> bool:
    return (o.kind in ("insight", "rule", "skill") and o.visibility != "private"
            and o.mode not in ("quarantine", "refuted", "withdrawn", "merged", "archived") and "knowledge_kind:code" not in o.hashtags)


def needs(o: g.MemoryObject) -> dict:
    from . import classify, qa_pairs
    qa_v = ((o.data.get("qa") or {}).get("version") or {})
    pairs = qa_v.get("prompt") not in (qa_pairs.prompt_sha256(), AGENT_PROMPT) or qa_v.get("schema") != qa_pairs.SCHEMA_VERSION
    labels = ((o.data.get("class") or {}).get("version") or {}).get("prompt") != classify.prompt_hash()
    tags = not any(t.startswith(TRIGGER_KINDS) for t in o.hashtags)
    return {"pairs": pairs, "labels": labels, "tags": tags}


def request(o: g.MemoryObject, instance: str) -> str | None:
    """The short request appended to vi_object, or None (nothing missing, not eligible, or asked recently)."""
    from . import classify, qa_pairs
    if not _eligible(o):
        return None
    n = needs(o)
    if not any(n.values()):
        return None
    k = (instance, o.id)
    if time.monotonic() - _asked.get(k, -1e9) < REQUEST_COOLDOWN_S:
        return None
    _asked[k] = time.monotonic()
    parts = []
    if n["pairs"]:
        parts.append(f"pairs: {qa_pairs.PAIRS} question/answer pairs in English, grounded in the text above (each question names its subject, "
                     "never 'the document'; every name and number of an answer occurs in the text)")
    if n["labels"]:
        parts.append(f"answers: one of {', '.join(classify.ANSWERS)} (what question it answers); memory: one of {', '.join(classify.MEMORY)}")
    parts.append("tags: up to 10 kind:value for the hosts, services, products and topics the text is about"
                 + (" (it has none)" if n["tags"] else " (only missing ones)"))
    return ("  QUALIFY (optional, once, ~300 tokens): this object lacks " + " and ".join(k for k, v in n.items() if v) +
            f". Since its text is in your context, you can send vi_qualify(id=\"{o.id}\", " + "; ".join(parts) + "). "
            "If you notice it is right or wrong, vi_confirm / vi_refute it with your evidence — your own objects too (a self-review counts half).")


def apply(oid: str, args: dict, instance: str, family: str | None = None) -> dict:
    """Check and store what an agent sent. Raises ValueError with the reason when nothing can be stored."""
    from . import classify, qa_pairs
    o = g.load(oid)
    if o is None:
        raise ValueError(f"{oid} does not exist")
    if not _eligible(o):
        raise ValueError(f"{oid} cannot be qualified (kind {o.kind}, mode {o.mode}, visibility {o.visibility})")
    out: dict = {"id": o.id}
    who = f"agent@{instance}"
    model = f"agent:{family or instance}"
    if args.get("pairs") is not None:
        doc = classify.document_of(o)
        kept, rejected = qa_pairs.select(list(args.get("pairs") or [])[:12], doc)
        train, probe, close = qa_pairs.split_probe(o.id, kept)
        if close:
            rejected["too close to the probe"] = rejected.get("too close to the probe", 0) + close
        res = {"state": "ok" if kept else "empty", "pairs": train, "probe": probe, "rejected": rejected,
               "content_sha256": qa_pairs.content_sha256(doc), "truncated": False, "chars": len(doc),
               "version": {"model": model, "prompt": AGENT_PROMPT, "schema": qa_pairs.SCHEMA_VERSION},
               "checked": "fixed rules (qa_pairs.select); no model support check"}
        if kept and classify.store_pairs(o, res, who):
            out["pairs"] = {"kept": len(train), "probe": bool(probe), "rejected": rejected}
        else:
            out["pairs"] = {"kept": 0, "rejected": rejected, "note": "nothing stored — fix the rejected reasons and send again"}
    if args.get("answers") is not None or args.get("memory") is not None:
        a, m = str(args.get("answers") or ""), str(args.get("memory") or "")
        if a not in classify.ANSWERS or m not in classify.MEMORY:
            raise ValueError(f"answers must be one of {classify.ANSWERS} and memory one of {classify.MEMORY}")
        lab = {"answers": a, "memory": m, "zweite": None, "zustand": "agent",
               "version": {"modell": model, "prompt": classify.prompt_hash(), "schema": classify.SCHEMA_VERSION}}
        classify.apply(g.load(o.id), lab, who)
        out["labels"] = {"answers": a, "memory": m}
    if args.get("tags"):
        added, refused = [], {}
        for t in [str(x).strip().lower() for x in list(args.get("tags") or [])[:MAX_TAGS]]:
            cur = g.load(o.id)
            if t in cur.hashtags:
                continue
            try:
                cur.tag([t], [], who, "vi_qualify: tagged by the reading agent")
                added.append(t)
            except ValueError as e:
                refused[t] = str(e)[:120]
        out["tags"] = {"added": added, **({"refused": refused} if refused else {})}
    if out.get("pairs", {}).get("kept"):
        classify._pairs_queue_table()  # created on first use; a fresh memory may not have it yet
        with g.lock:
            g.db().execute("DELETE FROM pairs_queue WHERE id = ?", (o.id,))
            g.db().commit()
    return out
