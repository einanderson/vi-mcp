"""Qualification on access (0.69.2): an object that is searched for or read and whose hashtags are incomplete is completed at that
moment, not only when sleep reaches it.

Why (the user, 17.11.2026: "wir sollten Datensätze, die unvollständige Tags haben, beim Aufrufen und Lesen dann noch Tags vergeben, so
dass die DB-Objekte on the fly qualifiziert werden"): sleep completes documentation step by step (6b: at most 3,000 objects per sleep,
QA pairs 8i only a handful) — after the import of 16,000 pages most of them would wait days for their triggers, while the ones agents
actually look at are the ones that matter. What is accessed migrates first; nothing runs a language model in the request.

Two completions, both cheap and deterministic:
1. labels answers:/memory: of the current classification version — the student model at once where it is sure, otherwise the object is
   put in front of sleep's queue (classify.on_search, 0.62.0);
2. host:/topic: triggers of documentation objects (knowledge_kind company-docs, external-docs, manual) whose derived_triggers are
   missing or of an older version — derived with the rules of sleep step 6b (doc_triggers.refresh), filler topics taken off again.
At most PER_CALL objects per access, never raises: a search or a read must not fail because of qualification.
"""
from __future__ import annotations

PER_CALL = 10


def on_access(objs, who: str = "vi", how: str = "search") -> dict:
    from . import classify, doc_triggers
    out = {"labels": {}, "triggers_refreshed": 0, "tags_changed": 0}
    objs = [o for o in objs if o is not None][:PER_CALL]
    try:
        out["labels"] = classify.on_search(objs)
    except Exception:  # noqa: BLE001
        pass
    for o in objs:
        try:
            if o.visibility != "private" and o.mode not in ("quarantine", "refuted", "withdrawn", "archived") and doc_triggers.stale(o):
                changed = doc_triggers.refresh(o, who, f"on {how}")
                out["triggers_refreshed"] += 1
                out["tags_changed"] += len(changed)
        except Exception:  # noqa: BLE001
            continue
    return out
