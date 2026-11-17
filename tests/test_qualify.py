"""0.69.2: objects with incomplete hashtags are completed when they are searched for or read (qualify.on_access)."""
from vi_mcp import doc_triggers, memory as g, qualify, tools


def _known_world():
    g.new_object("insight", "web1 runs nginx", ["host:web1.example.org"], {"source": "t"}, instance="a")
    for i in range(3):
        g.new_object("insight", f"backup fact {i}", ["topic:backup"], {"source": "t"}, instance="a")
    doc_triggers._cache["at"] = 0.0


def _doc(text, content, extra=()):
    return g.new_object("insight", text, ["knowledge_kind:company-docs", "source_kind:wiki", *extra], {"source": "wiki", "content": content},
                        instance="importer", visibility="internal")


def test_a_read_document_gets_its_triggers_at_once(gd):
    _known_world()
    d = _doc("Wiki IT: Backup of the web servers", "Run the job on web1.example.org every night.")
    assert doc_triggers.stale(d)
    tools.call_tool("vi_object", {"id": d.id}, "a", False)
    o = g.load(d.id)
    assert "host:web1.example.org" in o.hashtags and "topic:backup" in o.hashtags
    assert o.data["derived_triggers"]["at"] == "on read" and not doc_triggers.stale(o)


def test_a_searched_document_is_completed_and_filler_topics_go(gd):
    _known_world()
    d = _doc("Wiki IT: Backup for the old servers", "nightly backup job", extra=["topic:for"])
    d.data["derived_triggers"] = {"version": 1, "host": [], "topic": ["for"]}
    g.save(d)
    tools.call_tool("vi_recall", {"query": "backup old servers"}, "a", False)
    o = g.load(d.id)
    assert "topic:for" not in o.hashtags and "topic:backup" in o.hashtags and o.data["derived_triggers"]["at"] == "on search"


def test_a_measurement_does_not_qualify_and_current_objects_are_left_alone(gd):
    _known_world()
    d = _doc("Wiki IT: Backup of the web servers", "web1.example.org")
    tools.call_tool("vi_recall", {"query": "backup web servers", "reinforce": False}, "a", False)
    assert doc_triggers.stale(g.load(d.id))  # reinforce=false only reads (0.31.0)
    r = qualify.on_access([g.load(d.id)], "t", "read")
    assert r["triggers_refreshed"] == 1
    assert qualify.on_access([g.load(d.id)], "t", "read")["triggers_refreshed"] == 0  # done once, not on every access
    note = g.new_object("insight", "web1.example.org backup note", ["topic:x"], {"source": "t"}, instance="a")
    assert qualify.on_access([note], "t", "read")["triggers_refreshed"] == 0  # only documentation gets derived triggers
