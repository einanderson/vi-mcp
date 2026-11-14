"""Portable VIL object filters shared by the memory and frozen machine worlds."""
from __future__ import annotations
import re

TAG = re.compile(r"^[a-z][a-z0-9_-]{0,30}:[^\s()]+$")

def is_condition(tok, kinds):
    return bool(TAG.match(tok)) and (kinds is None or tok.split(":", 1)[0] in kinds)


def parse_filter(text: str, kinds: frozenset[str] | None = None) -> tuple[list[list[str]], list[str], list[str]]:
    """Filter text → (AND groups of OR tags, exclusion tags, free words). `(a OR b) c !d word`. Pure when `kinds` is given or None.
    0.26.0: a word with a colon is a hashtag condition only in the form kind:value (TAG) and, if `kinds` is given, with a kind that occurs
    in the memory — otherwise it is free text. Before, "Hosts: 62.153.216.33" in a question emptied the whole search silently (reported by
    the AIKB benchmark session, 10.11.2026)."""
    groups, remove, words = [], [], []
    for m in re.finditer(r"\(([^()]*)\)|(!?\S+)", text or ""):
        if m.group(1) is not None:
            old = [t for t in re.split(r"\s+OR\s+|\s+", m.group(1).strip()) if t and t != "OR"]
            if old:
                groups.append(old)
            continue
        tok = m.group(2)
        if tok.upper() == "OR":
            continue
        if tok.startswith("!"):
            remove.append(tok[1:])
        elif ":" in tok and is_condition(tok, kinds):
            groups.append([tok])
        else:
            words.append(tok)
    return groups, remove, words


def matches(obj, query, known):
    groups, remove, words = parse_filter(query, known)
    tags = {f"{k}:{v}" for k, vals in obj["tags"].items() for v in vals}
    def has(tag):
        return any(t.startswith(tag[:-1]) for t in tags) if tag.endswith("*") else tag in tags
    return all(any(has(t) for t in group) for group in groups) and not any(has(t) for t in remove) and all(
        w.lower() in (obj.get("text", "") + " " + " ".join(tags)).lower() for w in words)


def object_data(objects, filters, edges=()):
    known = {k for o in objects for k in o["tags"]} | {"art", "host", "thema"}
    members = [[o["id"] for o in objects if matches(o, f, known)] for f in filters]
    ids = {i for m in members for i in m}
    return {"filters": list(filters), "members": members, "objects": [o for o in objects if o["id"] in ids],
            "edges": [e for e in edges if e[0] in ids and e[1] in ids],
            "counts": {"cut": 0, "not_visible": 0, "excluded": 0, "via_edge": 0}, "limit": len(objects)}
