"""Credentials in documentation, checked again (0.69.3).

Why (production 17.11.2026): an imported Confluence page held recovery answers and access data in clear text — the import scrubbed
only what `anonymize.without_secrets` knows (key=value forms, keys, tokens); `anonymize.strip_credentials`, the stronger check for
documents (password columns, "das Passwort lautet …", the line after "Passwort:"), existed but was called nowhere, and it missed
single-"|" tables, security-question answers and "Kennwort Xy7!…" without a separator. The user: "schärfe die Geheimnis-Prüfung nach".

Documentation objects (doc_triggers.DOC_TAGS) are checked with strip_credentials of version anonymize.CREDENTIALS_VERSION:
- at import (doc_import), on access (qualify.on_access), and in sleep for everything not yet checked (SWEEP_PER_SLEEP per sleep);
- the long-term archive too (archived pages keep their raw text there): `vi-mcp secrets sweep` checks both completely.
A replaced value becomes <geheimnis>; the object records data.secrets_checked = version and a change "credentials replaced: N"
(the count only — never the value). The old text is not kept in the object's history (a change records who/what/why, not content).
"""
from __future__ import annotations

import json
import zlib

from . import anonymize, archive, memory as g

SWEEP_PER_SLEEP = 5000


def _is_doc(tags) -> bool:
    from . import doc_triggers
    return any(t in tags for t in doc_triggers.DOC_TAGS)


def check(o: g.MemoryObject, who: str = "vi", why: str = "secret check") -> int:
    """Check one documentation object; save it (marked as checked) and return how many values were replaced."""
    if (o.data.get("secrets_checked") or 0) >= anonymize.CREDENTIALS_VERSION:
        return 0
    text, n1 = anonymize.strip_credentials(o.text or "")
    content, n2 = anonymize.strip_credentials(str(o.data.get("content") or "")) if o.data.get("content") else ("", 0)
    if n1 or n2:
        o.text = text
        if o.data.get("content"):
            o.data["content"] = content
        o.change(who, f"credentials replaced: {n1 + n2}", f"{why}: strip_credentials v{anonymize.CREDENTIALS_VERSION}")
    o.data["secrets_checked"] = anonymize.CREDENTIALS_VERSION
    g.save(o)
    return n1 + n2


def pending(limit: int) -> list[str]:
    from . import doc_triggers
    with g.lock:
        return [r[0] for r in g.db().execute(
            f"SELECT id FROM object WHERE id IN (SELECT object_id FROM tag WHERE tag IN ({','.join('?' * len(doc_triggers.DOC_TAGS))})) "
            "AND mode != 'archived' AND COALESCE(CAST(json_extract(data, '$.secrets_checked') AS REAL), 0) < ? LIMIT ?",
            (*doc_triggers.DOC_TAGS, anonymize.CREDENTIALS_VERSION, int(limit))).fetchall()]


def run(limit: int = SWEEP_PER_SLEEP, who: str = "sleep") -> dict:
    out = {"checked": 0, "objects_changed": 0, "values_replaced": 0}
    for oid in pending(limit):
        o = g.load(oid)
        if o is None:
            continue
        k = check(o, who, "sweep")
        out["checked"] += 1
        out["objects_changed"] += bool(k)
        out["values_replaced"] += k
    out["open"] = len(pending(1_000_000))
    return out


def run_archive(who: str = "cli") -> dict:
    """The archived documentation (raw JSON in the archive database): strip_credentials on its text and content, rows rewritten."""
    out = {"checked": 0, "entries_changed": 0, "values_replaced": 0}
    with g.lock:
        rows = archive.db().execute("SELECT id, raw FROM object").fetchall()
    for oid, raw in rows:
        blob = json.loads(zlib.decompress(raw))
        if not _is_doc(blob.get("hashtags") or []):
            continue
        out["checked"] += 1
        data = blob.get("data", blob.get("daten")) or {}
        text, n1 = anonymize.strip_credentials(blob.get("text") or "")
        key = "content" if "content" in data else "inhalt"
        content, n2 = anonymize.strip_credentials(str(data.get(key) or "")) if data.get(key) else ("", 0)
        if n1 or n2:
            blob["text"] = text
            if data.get(key):
                data[key] = content
            new = zlib.compress(json.dumps(blob, ensure_ascii=False).encode())
            with g.lock:
                archive.db().execute("UPDATE object SET raw = ? WHERE id = ?", (new, oid))
                archive.db().commit()
            out["entries_changed"] += 1
            out["values_replaced"] += n1 + n2
    return out


def main(argv: list[str]) -> int:
    """vi-mcp secrets sweep — check all documentation and the archive with the current strip_credentials (counts only, no values)."""
    from . import config, vector
    if not argv or argv[0] != "sweep":
        print(main.__doc__)
        return 2
    c = config.load()
    g.open_db(c.db)
    vector.config = c.embed
    print(json.dumps({"memory": run(10_000_000, "cli"), "archive": run_archive("cli")}))
    return 0
