"""0.68.1: vi_learn(supersedes=[…]) — a re-import in a new shape (one page as one object instead of section parts) retires the instance's
own old objects: archived with a reason naming the new one, linked to it, restorable; nothing disappears silently."""
import pytest

from vi_mcp import archive, memory as g, tools
from vi_mcp.tools import Rejected

BASE = {"hashtags": ["topic:docs", "source_kind:confluence"], "source": "https://wiki.example.org/page/1 · version 3"}


def _learn(instance, text, **extra):
    out = tools.call_tool("vi_learn", {**BASE, "text": text, **extra}, instance, False)
    return out.split()[1], out


def test_supersedes_archives_the_own_parts_links_and_records_them(gd):
    p1, _ = _learn("a", "Page X (part 1/2): the first section")
    p2, _ = _learn("a", "Page X (part 2/2): the second section")
    new, out = _learn("a", "Page X: the whole page", content="## First\none\n## Second\ntwo", key="wiki:x", supersedes=[p1, p2])
    assert "superseded 2 object(s)" in out
    for oid in (p1, p2):
        o = g.load(oid)
        assert o.mode == "archived" and f"superseded by {new}" in o.data["reason"]
        assert archive.content(oid)["text"].startswith("Page X (part")  # the full old content is kept, restorable
        assert any(b == oid and "supersedes" in r for b, r in g.load(new).edges())
    assert sorted(g.load(new).data["supersedes"]) == sorted([p1, p2])
    assert g.load(new).mode == "active"


def test_another_instances_object_refuses_the_whole_call_before_anything_is_stored(gd):
    theirs, _ = _learn("b", "Someone else's section text")
    before = g.db().execute("SELECT COUNT(*) FROM object").fetchone()[0]
    with pytest.raises(Rejected, match="belongs to instance b"):
        _learn("a", "My whole page about it", supersedes=[theirs])
    assert g.db().execute("SELECT COUNT(*) FROM object").fetchone()[0] == before
    assert g.load(theirs).mode == "active"


def test_unknown_ids_too_many_ids_and_wrong_kinds_are_refused(gd):
    with pytest.raises(Rejected, match="does not exist"):
        _learn("a", "A page that replaces nothing real", supersedes=["insight:doesnotexist"])
    with pytest.raises(Rejected, match="at most"):
        _learn("a", "A page that replaces too much", supersedes=[f"insight:{i}" for i in range(tools.SUPERSEDES_MAX + 1)])
    exp = g.new_object("experience", "ran a command", ["tool:Bash"], {}, instance="a")
    with pytest.raises(Rejected, match="only insight"):
        _learn("a", "A page that replaces an experience", supersedes=[exp.id])


def test_an_already_archived_part_is_skipped_and_a_keyed_update_can_supersede(gd):
    p1, _ = _learn("a", "Page Y (part 1/2): text one")
    p2, _ = _learn("a", "Page Y (part 2/2): text two")
    archive.store_in_archive(g.load(p2), "archived earlier")
    new, out = _learn("a", "Page Y: first version", key="wiki:y")
    same, out = _learn("a", "Page Y: whole page", key="wiki:y", supersedes=[p1, p2])
    assert same == new and "updated" in out and "superseded 1 object(s)" in out
    assert g.load(p1).mode == "archived" and g.load(new).data["supersedes"] == [p1]


def test_the_archive_opens_cleanly_when_its_object_table_has_no_data_column(gd):
    """0.68.2: the 0.67 name check asked the archive's object table for a data column it never has — the first archive access after
    every server start failed with "no such column: data" (17.11.2026, production)."""
    archive.close_db()
    with g.lock:
        con = archive.db()  # the archive's object table exists (English) and is empty, as on production after 0.67.0
    assert "data" not in [r[1] for r in con.execute("PRAGMA table_info(object)")]
    archive.close_db()
    p, _ = _learn("a", "Page Z (part 1/1): some text")
    new, out = _learn("a", "Page Z: the whole page", supersedes=[p])  # the first archive access of a fresh process
    assert "superseded 1 object(s)" in out and g.load(p).mode == "archived"
