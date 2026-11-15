"""0.67.0: the stored names are English. A database of 0.66.1 (tests/fixtures, synthetic) is migrated in place when it is opened:
backup copy first, every table and column renamed, the counts recorded — and afterwards no German stored name is left."""
import hashlib
import os
import sqlite3
from pathlib import Path

from vi_mcp import db as dbmod
from vi_mcp import db_migrate
from vi_mcp import logbook as lb
from vi_mcp import memory as g
from vi_mcp import names

FIXTURES = Path(__file__).parent / "fixtures"
PG = os.environ.get("VI_MCP_TEST_PG")
GERMAN_TABLES = set(names.TABLES) | set(names.LOGBOOK_TABLES)
GERMAN_COLUMNS = {old for m in names.COLUMNS.values() for old in m}


def _load(sql_file: str, target: Path) -> dict[str, int]:
    """The fixture into a SQLite file (and, when the suite runs against PostgreSQL, into the schema the test hooks give that file).
    Returns the row count per table before the migration."""
    con = sqlite3.connect(target)
    con.executescript((FIXTURES / sql_file).read_text(encoding="utf-8"))
    counts = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
              for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'objekt_fts_%'")}
    con.close()
    if PG:
        prefix = "t_" if sql_file.startswith("memory") else "l_"
        db_migrate.copy(target, PG, prefix + hashlib.md5(str(target).encode()).hexdigest()[:20], say=lambda *_a: None)
    return counts


def _english(con) -> tuple[set[str], dict[str, set[str]]]:
    tables = names._tables(con)
    return tables, {t: names._columns(con, t) for t in tables if not t.startswith("object_fts_")}


def _assert_no_german(con, before: dict[str, int], mapping: dict[str, str]) -> None:
    tables, columns = _english(con)
    assert not tables & GERMAN_TABLES, tables & GERMAN_TABLES
    left = {f"{t}.{c}" for t, cols in columns.items() for c in cols & set(names.COLUMNS.get(t, {}))}
    assert not left, left
    for old, n in before.items():  # nothing lost: every row is still there under the English name
        new = mapping.get(old, old)
        assert con.execute(f"SELECT COUNT(*) FROM {new}").fetchone()[0] == n, (old, new)


def test_the_mapping_is_one_to_one():
    for m in [names.TABLES, names.LOGBOOK_TABLES, names.INDEXES, *names.COLUMNS.values()]:
        assert len(set(m.values())) == len(m), m  # reversible: no two German names become one English name
        assert not set(m) & set(m.values()) - {k for k, v in m.items() if k == v}
    assert not set(names.TABLES.values()) & set(names.LOGBOOK_TABLES.values())
    reserved = {"from", "on", "group", "join", "order", "values", "select", "where", "table", "index"}
    every = set(names.TABLES.values()) | set(names.LOGBOOK_TABLES.values()) | {c for m in names.COLUMNS.values() for c in m.values()}
    assert not every & reserved


def test_a_0_66_1_memory_is_migrated_in_place_with_a_backup(tmp_path):
    p = tmp_path / "vi.db"
    before = _load("memory_0_66_1.sql", p)
    assert "objekt" in before and before["objekt"] >= 20
    g.open_db(p)
    try:
        with g.lock:
            _assert_no_german(g.db(), before, names.TABLES)
            rep = {cls: n for _s, cls, n, _t, _b in names.report(g.db())}
            backup = names.report(g.db())[0][4]
        assert rep["tables"] == len(before) - sum(1 for t in before if t in ("tag", "vi_index", "net_peer_removed"))
        assert rep["columns"] > 100
        if not PG:
            copy = Path(backup)
            assert copy.exists() and oct(copy.stat().st_mode & 0o777) == "0o600"
            old = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)  # the backup is the untouched 0.66.1 state
            assert old.execute("SELECT COUNT(*) FROM objekt").fetchone()[0] == before["objekt"]
            old.close()
        else:
            assert backup.startswith("schema t_") and backup.endswith("_pre_0_67_0")
        assert g.load(g.recall("thema:backup", viewer=None)[0][1].id).text == "Backups on db01.example run at 02:00"
        assert "stored names: English — migrated" in db_migrate._names_lines(g.db(), "object")
    finally:
        g.close_db()
    g.open_db(p)  # idempotent: the second start finds nothing to do and writes no second report
    try:
        with g.lock:
            assert names.report(g.db())[0][4] == backup and not names._pending_schema(g.db(), names.TABLES)
    finally:
        g.close_db()


def test_a_new_memory_needs_no_migration(gd):
    with g.lock:
        assert names.report(g.db()) == [] and not names._pending_schema(g.db(), names.TABLES)
        tables, columns = _english(g.db())
    assert not tables & GERMAN_TABLES
    assert "stored names: English (created by 0.67.0" in db_migrate._names_lines(g.db(), "object")


def test_a_0_66_1_logbook_is_migrated(tmp_path):
    p = tmp_path / "netz.db"
    before = _load("logbook_0_66_1.sql", p)
    seed = tmp_path / "seed"
    seed.write_bytes(b"\x01" * 32)
    log = lb.Logbook(p, seed, [])
    _assert_no_german(log.db, before, names.LOGBOOK_TABLES)
    assert log.db.execute("SELECT failures, last_error FROM peer").fetchone()[0] == 1
    assert log.db.execute("SELECT type FROM entry").fetchone()[0] == "objekt"  # the wire form of a signed entry stays as it was signed
    assert {cls for _s, cls, *_r in names.report(log.db)} == {"tables", "columns"}


def test_the_archive_is_migrated(tmp_path):
    from vi_mcp import archive
    p = tmp_path / "vi.db"
    g.open_db(p)
    g.close_db()
    if not PG:
        a = sqlite3.connect(tmp_path / "vi-lager.db")
        a.executescript("DROP TABLE IF EXISTS object; DROP TABLE IF EXISTS conversation_raw;"
                        "CREATE TABLE objekt (id TEXT PRIMARY KEY, art TEXT NOT NULL, instanz TEXT NOT NULL, sicht TEXT NOT NULL, eingelagert TEXT NOT NULL, "
                        "modus_vorher TEXT NOT NULL, grund TEXT NOT NULL, roh BLOB NOT NULL);"
                        "INSERT INTO objekt VALUES ('erkenntnis:a', 'erkenntnis', 'anna', 'geteilt', '2026-11-01', 'aktiv', 'old', X'00');"
                        "CREATE TABLE gespraech_roh (serial TEXT PRIMARY KEY, eingelagert TEXT NOT NULL, roh BLOB NOT NULL, laenge INTEGER NOT NULL);")
        a.close()
    g.open_db(p)
    try:
        con = archive.db()
        tables, columns = _english(con)
        assert {"object", "conversation_raw"} <= tables and not tables & GERMAN_TABLES
        assert columns["object"] == {"id", "kind", "instance", "visibility", "archived", "mode_before", "reason", "raw"}
        if not PG:
            assert con.execute("SELECT kind, mode_before FROM object").fetchone() == ("erkenntnis", "aktiv")
    finally:
        g.close_db()


def test_a_full_disk_refuses_before_changing_anything(tmp_path, monkeypatch):
    if PG:
        return
    p = tmp_path / "vi.db"
    _load("memory_0_66_1.sql", p)
    monkeypatch.setattr(names.shutil, "disk_usage", lambda _p: type("U", (), {"free": 10})())
    try:
        g.open_db(p)
        raise AssertionError("expected a refusal")
    except RuntimeError as e:
        assert "nothing was changed" in str(e)
    con = sqlite3.connect(p)
    assert con.execute("SELECT COUNT(*) FROM objekt").fetchone()[0] > 0  # still the German schema
    con.close()
    assert dbmod.is_url("x") is False
