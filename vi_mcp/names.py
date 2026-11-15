"""Stored names: German (up to 0.66.x) → English (0.67.0). The one table of every renamed stored name.

Used by
- the migration of an existing database when it is opened (`rename_schema`): tables and columns are renamed in place, values,
  hashtags and data keys are rewritten — nothing is copied, nothing is dropped;
- the reading of old names that still arrive: tool input of clients registered before 0.67.0, files written before (index files,
  archives), and the network;
- the network boundary (`p2p`, `net_score`, `wiki`, `forum`): signed logbook entries and content hashes are the German wire form of
  every server before 0.67.0. They cannot be renamed without breaking every signature and every hash, so the wire form stays German
  and is translated here on the way out (`wire_*`) and on the way in;
- the code check `german_stored_names`, which reports a German stored name used outside this module and the wire boundary.

Every German name maps to exactly one English name, and no two German names map to the same English name within one mapping
(checked by tests/test_names.py), so the translation is reversible where the wire needs the old form.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .log_output import log

#: tables of the memory database (and of its long-term archive, which uses objekt and gespraech_roh)
TABLES: dict[str, str] = {
    "objekt": "object", "kante": "edge", "gespraech_roh": "conversation_raw", "objekt_fts": "object_fts",
    "werkzeug_register": "tool_register", "klasse_warteschlange": "class_queue", "zentrum": "centre",
    "inhalt_gruppe": "content_cluster", "inhalt_gruppe_name": "content_cluster_name", "suche_log": "search_log",
    "beitrag_aus": "post_out", "beitrag_ein": "post_in", "bild": "image",
    "logik_gewicht": "logic_weight", "logik_bewertung": "logic_rating", "logik_frage": "logic_question",
    "p2p_inhalt": "p2p_content", "p2p_halter": "p2p_holder", "p2p_bewertung": "p2p_rating", "p2p_gesehen": "p2p_seen",
    "netz_token": "net_token", "netz_widerruf": "net_revocation", "p2p_netz": "p2p_net", "p2p_grab": "p2p_tombstone",
    "netz_stand": "net_state", "netz_schalter": "net_switch", "netz_suche": "net_search",
    "relevanz_urteil": "relevance_judgment", "selbst_version": "self_version", "nutzung": "usage", "vektor": "vector",
    "seite_rev": "page_rev", "seite_eigen": "page_own", "seite_widerruf": "page_revocation",
    "vil_gewicht": "vil_weight", "vil_gewicht_netz": "vil_weight_net",
    "p2p_stand": "p2p_state",  # a table of 0.39 that no code reads any more; renamed, not dropped (nothing disappears silently)
}

#: tables of a network logbook (the separate database of the network service vi-p2p)
LOGBOOK_TABLES: dict[str, str] = {
    "eintrag": "entry", "inhalt": "content", "grab": "tombstone", "gabelung": "fork", "verstoss": "violation", "stand": "state",
    "beitritt": "joining", "identitaet": "identity",
}

#: columns per table (English table name → {German column: English column}). Columns whose name is already English are not listed.
COLUMNS: dict[str, dict[str, str]] = {
    "object": {"art": "kind", "zeit": "ts", "zuletzt": "last_at", "daten": "data", "wichtigkeit": "importance", "bestaetigt": "confirmed",
               "widerlegt": "refuted", "zugriffe": "accesses", "modus": "mode", "aenderungen": "changes", "gespraech": "conversation",
               "instanz": "instance", "sicht": "visibility",
               # the long-term archive's objekt table (archive.py)
               "eingelagert": "archived", "modus_vorher": "mode_before", "grund": "reason", "roh": "raw"},
    "tag": {"objekt_id": "object_id"},
    "edge": {"grund": "reason", "zeit": "ts"},
    "conversation_raw": {"roh": "raw", "laenge": "length", "eingelagert": "archived"},
    "vi_index": {"art": "kind", "beschreibung": "description", "erstellt": "created", "quelle": "source"},
    "tool_register": {"zeit": "ts", "instanz": "instance"},
    "class_queue": {"zeit": "ts"},
    "centre": {"summe": "sum"},
    "content_cluster": {"gruppe": "cluster"},
    "content_cluster_name": {"gruppe": "cluster"},
    "search_log": {"zeit": "ts", "instanz": "instance", "gespraech": "conversation", "anfrage": "query", "suchart": "search_kind",
                   "treffer": "hits", "geoeffnet": "opened"},
    "post_out": {"netz": "net", "art": "kind", "faden": "thread", "eltern": "parent", "instanz": "instance", "daten": "data", "zeit": "ts"},
    "post_in": {"netz": "net", "art": "kind", "faden": "thread", "eltern": "parent", "autor": "author", "netz_id": "net_id",
                "loesch_hash": "delete_hash", "empfangen": "received", "entschieden": "decided"},
    "image": {"breite": "width", "hoehe": "height", "zeit": "ts", "roh": "raw"},
    "logic_weight": {"regel": "rule", "ja": "yes", "nein": "no", "zeit": "ts"},
    "logic_rating": {"zeit": "ts", "programm_hash": "program_hash", "schluss": "conclusion", "richtig": "correct", "instanz": "instance"},
    "logic_question": {"zeit": "ts", "instanz": "instance", "frage": "question", "programm": "program", "programm_hash": "program_hash",
                       "zustand": "state", "antworten": "answers", "runden": "rounds"},
    "p2p_content": {"inhalt_sha": "content_sha"},
    "p2p_holder": {"inhalt_sha": "content_sha", "netz": "net", "netz_id": "net_id", "zeit": "ts"},
    "p2p_rating": {"inhalt_sha": "content_sha", "netz": "net", "gezeigt": "shown", "ja": "yes", "nein": "no", "bestaetigt": "confirmed",
                   "widerlegt": "refuted", "zeit": "ts"},
    "p2p_seen": {"herausgeber": "publisher", "anzahl": "count"},
    "p2p_state": {"zuletzt": "last_at"},
    "net_token": {"instanz": "instance", "zeit": "ts"},
    "net_revocation": {"instanz": "instance", "netz_id": "net_id", "zeit": "ts", "grund": "reason"},
    "p2p_net": {"netz_id": "net_id", "loesch_hash": "delete_hash"},
    "p2p_tombstone": {"netz_id": "net_id", "zeit": "ts"},
    "net_state": {"netz": "net", "zeit": "ts", "daten": "data"},
    "net_switch": {"netz": "net", "an": "enabled", "zeit": "ts", "wer": "who", "grund": "reason"},
    "net_search": {"zeit": "ts", "netz": "net", "art": "kind", "gefragt": "asked", "antworten": "answers", "nicht_erreichbar": "unreachable",
                   "treffer": "hits"},
    "net_peer_removed": {"time": "ts"},  # 0.66.1 named it time; one word for one thing: ts like every other table
    "relevance_judgment": {"suche": "search", "zeit": "ts", "suche_zeit": "search_ts", "anfrage": "query", "suchart": "search_kind",
                           "objekt_id": "object_id", "rang": "rank", "note": "grade", "grund": "reason"},
    "self_version": {"zeit": "ts", "wer": "who", "bereich": "area", "satz": "sentence", "grund": "reason", "belege": "evidence",
                     "vorschlag": "proposal"},
    "usage": {"zeit": "ts", "instanz": "instance", "werkzeug": "tool", "ausgang": "outcome", "treffer": "hits", "suchart": "search_kind",
              "leer_grund": "empty_reason"},
    "vector": {"modell": "model", "zeit": "ts", "roh": "raw", "verglichen": "compared"},
    "page_rev": {"netz": "net", "seite": "page", "eltern": "parents", "instanz": "instance", "daten": "data", "netz_id": "net_id",
                 "loesch_hash": "delete_hash", "empfangen": "received", "geprueft": "reviewed", "entschieden": "decided", "eigen": "own"},
    "page_own": {"netz": "net", "instanz": "instance", "schluessel": "key"},
    "page_revocation": {"netz": "net", "netz_id": "net_id", "zeit": "ts", "grund": "reason"},
    "vil_weight": {"wort": "word", "frageart": "question_kind", "ja": "yes", "nein": "no", "zeit": "ts"},
    "vil_weight_net": {"netz": "net", "ja": "yes", "nein": "no", "zeit": "ts"},
    # logbook database
    "entry": {"typ": "type", "netz_id": "net_id", "loesch_hash": "delete_hash", "zeit": "ts", "kopf": "header"},
    "content": {"daten": "data"},
    "tombstone": {"netz_id": "net_id", "zeit": "ts"},
    "fork": {"zeit": "ts"},
    "peer": {"adresse": "address", "quelle": "source", "zuletzt": "last_at", "fehler": "failures", "naechster": "next_at",
             "version_zeit": "version_ts", "fehlt_hier": "missing_here", "fehlt_dort": "missing_there"},
    "joining": {"ablauf": "expiry", "benutzt": "used"},
    "segment": {"von": "first_seq", "bis": "last_seq", "datei": "file"},
    "identity": {"zeit": "ts"},
}

#: indexes of the German schema → their English names. SQLite cannot rename an index, so the migration drops the old one and the code
#: creates it again under the English name (tag_tag keeps its name)
INDEXES: dict[str, str] = {
    "objekt_art_zeit": "object_kind_ts", "objekt_idx": "object_idx", "objekt_instanz": "object_instance", "objekt_fts_tsv": "object_fts_tsv",
    "beitrag_ein_netz_id": "post_in_net_id", "eintrag_netz": "entry_net", "eintrag_typ": "entry_type", "nutzung_zeit": "usage_ts",
    "p2p_halter_netz_id": "p2p_holder_net_id", "p2p_inhalt_oid": "p2p_content_oid", "seite_rev_netz_id": "page_rev_net_id",
    "seite_rev_seite": "page_rev_page", "suche_log_zeit": "search_log_ts", "vil_gewicht_hash": "vil_weight_hash",
}


def _tables(con: Any) -> set[str]:
    from . import db as dbmod
    if dbmod.backend(con) == "postgres":
        return {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()")}
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _columns(con: Any, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def rename_schema(con: Any, tables: dict[str, str]) -> list[str]:
    """Rename the German tables of `tables` (and every listed column of each) in place. Idempotent: a table or column already English
    is left alone; a database with both the German and the English table (should not exist) is refused, never merged.
    Returns what was renamed, for the log. The caller commits."""
    done: list[str] = []
    present = _tables(con)
    for old, new in tables.items():
        if old == new:  # a table whose name is already English: only its columns are renamed
            continue
        if old in present and new in present:
            raise RuntimeError(f"both {old} and {new} exist — a half-migrated database; nothing renamed, check it by hand")
        if old in present:
            con.execute(f"ALTER TABLE {old} RENAME TO {new}")
            done.append(f"{old}→{new}")
            present = (present - {old}) | {new}
    for table in {new for new in tables.values()} | (set(COLUMNS) & present):
        if table not in present or table not in COLUMNS or table.endswith("_fts"):
            continue
        cols = _columns(con, table)
        for old, new in COLUMNS[table].items():
            if old in cols and new not in cols:
                con.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
                done.append(f"{table}.{old}→{new}")
    if done:
        for ix in INDEXES:
            con.execute(f"DROP INDEX IF EXISTS {ix}")
    if done:
        log(f"migration 0.67.0: {len(done)} stored name(s) German → English ({', '.join(done[:8])}{' …' if len(done) > 8 else ''})")
    return done


#: the migration of 0.67.0 as recorded in each database it changed (table name_migration: one row per class)
STEP = "0.67.0"


def _pending_schema(con: Any, tables: dict[str, str]) -> bool:
    present = _tables(con)
    if any(old in present for old, new in tables.items() if old != new):
        return True
    return any(old in _columns(con, t) for t in set(COLUMNS) & present if not t.endswith("_fts") for old in COLUMNS[t])


def _backup(con: Any, target: Any, label: str) -> str:
    """A copy of the database before the first change. SQLite: <file>.pre-0.67.0.db next to it (0600); PostgreSQL: the schema
    <schema>_pre_0_67_0 with a copy of every table. An existing copy is kept — it is the older state. Refuses (RuntimeError) when the
    disk cannot hold the copy, before anything is changed."""
    from . import db as dbmod
    if dbmod.backend(con) == "postgres":
        dest = f"{con.schema}_pre_0_67_0"
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{dest}"')
        have = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = ?", (dest,))}
        for t in sorted(_tables(con) - have):
            con.execute(f'CREATE TABLE "{dest}"."{t}" AS TABLE "{con.schema}"."{t}"')
        con.commit()
        return f"schema {dest}"
    src = Path(str(target))
    dest_path = src.with_name(f"{src.name}.pre-0.67.0.db")
    if dest_path.exists():
        return f"{dest_path} (kept from an earlier start)"
    size = sum(f.stat().st_size for f in (src, Path(f"{src}-wal")) if f.exists())
    free = shutil.disk_usage(src.parent).free
    if free < size * 1.1:
        raise RuntimeError(f"{label}: the migration to English names needs a backup copy of {size // 2**20} MiB, but {src.parent} has only "
                           f"{free // 2**20} MiB free — nothing was changed; free space and start again")
    import os
    import sqlite3
    old_umask = os.umask(0o077)
    try:
        d = sqlite3.connect(str(dest_path))
        con.backup(d)
        d.close()
    finally:
        os.umask(old_umask)
    os.chmod(dest_path, 0o600)
    return str(dest_path)


def _record(con: Any, counts: dict[str, int], backup: str | None) -> None:
    from datetime import datetime, timezone
    con.execute("CREATE TABLE IF NOT EXISTS name_migration (step TEXT NOT NULL, class TEXT NOT NULL, n INTEGER NOT NULL, ts TEXT NOT NULL, "
                "backup TEXT, PRIMARY KEY (step, class))")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for cls, n in counts.items():
        con.execute("INSERT OR REPLACE INTO name_migration (step, class, n, ts, backup) VALUES (?, ?, ?, ?, ?)", (STEP, cls, n, ts, backup))


def report(con: Any) -> list[tuple[str, str, int, str, str | None]]:
    """What the migration changed in this database: (step, class, count, time, backup), empty if it never ran here."""
    if "name_migration" not in _tables(con):
        return []
    return con.execute("SELECT step, class, n, ts, backup FROM name_migration ORDER BY step, class").fetchall()


def migrate_db(con: Any, tables: dict[str, str], target: Any, label: str) -> dict[str, int]:
    """The migration of one database (label: memory, archive or a network's logbook) from German to English stored names, before the
    code creates its tables. Nothing to do → {} and nothing is written. Otherwise: backup copy first, then the rename, then the counts
    per class into the table name_migration and the log. The caller commits."""
    if not _pending_schema(con, tables):
        return {}
    backup = _backup(con, target, label)
    log(f"migration {STEP} ({label}): backup copy before renaming: {backup}")
    done = rename_schema(con, tables)
    counts = {"tables": sum(1 for d in done if "." not in d.split("→")[0]), "columns": sum(1 for d in done if "." in d.split("→")[0])}
    _record(con, counts, backup)
    log(f"migration {STEP} ({label}): " + ", ".join(f"{n} {cls}" for cls, n in counts.items()))
    return counts
