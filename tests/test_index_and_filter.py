"""0.69.8: searching one project finds only that project (the user, 17.11.2026: "Wenn man nur nach einem Projekt suchen will, dürfen
auch nur Projekteinträge gefunden werden. Das wäre dann eine AND-Verknüpfung."). index:<name> in any filter is a condition ANDed with
the rest — vi_recall, vi_drilldown, the candidates of the semantic search and vi_logic objects(…) alike."""
import pytest

from vi_mcp import memory as g, tools


def _two_projects():
    a = g.new_object("insight", "backup of the build host runs at night", ["topic:backup", "host:build1.example"], {"source": "t"},
                     instance="x", index="alpha")
    b = g.new_object("insight", "backup of the mail relay runs at night", ["topic:backup", "host:mail1.example"], {"source": "t"},
                     instance="x", index="beta")
    c = g.new_object("insight", "backup policy for every host", ["topic:backup"], {"source": "t"}, instance="x")
    return a, b, c


def _ids(out):
    import re
    return set(re.findall(r"\[(insight:[0-9a-f]+)\]", out))


def test_index_in_the_filter_is_an_and_condition(gd):
    a, b, c = _two_projects()
    assert {o.id for _s, o, _w in g.recall("topic:backup", limit=10, viewer=None)} == {a.id, b.id, c.id}
    assert {o.id for _s, o, _w in g.recall("index:alpha topic:backup", limit=10, viewer=None)} == {a.id}
    assert {o.id for _s, o, _w in g.recall("index:alpha backup night", limit=10, viewer=None)} == {a.id}  # with free text too
    assert g.recall("index:alpha index:beta topic:backup", limit=10, viewer=None) == []  # two projects ANDed: nothing
    assert set(g.ids_for_filter("index:beta topic:backup", viewer=None)) == {b.id}  # semantic candidates


def test_every_search_tool_obeys_it(gd):
    a, b, c = _two_projects()
    assert _ids(tools.call_tool("vi_recall", {"filter": "index:alpha backup", "semantic": False}, "x", False)) == {a.id}
    assert _ids(tools.call_tool("vi_recall", {"filter": "backup", "index": "alpha", "semantic": False}, "x", False)) == {a.id}
    drill = tools.call_tool("vi_drilldown", {"filter": "index:beta topic:backup"}, "x", False)
    assert b.id in drill and a.id not in drill and c.id not in drill


def test_vi_logic_objects_take_a_project_as_a_set(gd):
    pytest.importorskip("clingo")
    a, b, c = _two_projects()
    out = tools.call_tool("vi_logic", {"action": "ask", "source": 'in_alpha(O) <= (O in objects("index:alpha"))',
                                       "queries": [f'in_alpha("{a.id}")', f'in_alpha("{b.id}")']}, "x", False)
    assert f'in_alpha("{a.id}"): true' in out and f'in_alpha("{b.id}"): not_derived' in out
