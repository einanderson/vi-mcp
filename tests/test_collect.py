"""Collect mode of vi_recall (vi_mcp/collect.py): a question about everything of one entity gets the chunks of every object about it
under a token budget. Invented hosts and pages only."""
from vi_mcp import collect, tools, vector
from vi_mcp import memory as g

PAGE = "https://wiki.example.org/pages/100/host7"


def _page(gd):
    g.new_object("insight", "Wiki: host7.example.org – Hardware: (part 1/2): board X470", ["host:host7.example.org"],
                 {"source": f"{PAGE} · version 5", "content": "## Hardware\n\n| CPU | 16 cores |\n|---|---|\n| RAM | 128 GB |\n"},
                 reason="T", visibility="shared")
    g.new_object("insight", "Wiki: host7.example.org – Disks: (part 2/2): two NVMe", ["host:host7.example.org"],
                 {"source": f"{PAGE} · version 5", "content": "## Disks\n\nTwo NVMe in a ZFS mirror, 2 TB each.\n"},
                 reason="T", visibility="shared")
    g.new_object("insight", "Cluster overview: host7 and host8 form cluster C7", ["host:host7.example.org", "host:host8.example.org"],
                 {"source": "https://wiki.example.org/pages/200/cluster", "content": "## Members\n\nhost7, host8.\n"},
                 reason="T", visibility="shared")
    g.new_object("insight", "Printer driver install on desk12", ["host:desk12.example.org"],
                 {"source": "https://wiki.example.org/pages/300/printer", "content": "## Driver\n\nInstall the driver.\n"},
                 reason="T", visibility="shared")


def test_the_router_tells_collect_questions_from_single_facts(gd):
    _page(gd)
    assert "collect intent" in collect.detect("Welche Ausstattung hat host7?")["reason"]
    assert collect.detect("what hardware does host7 have")["entities"] == ["host:host7.example.org"]
    assert "the entity" in collect.detect("host7")["reason"]
    assert collect.detect("backup failed on host7 since yesterday") is None  # a single fact about the host
    assert collect.detect("which printer driver") is None                     # intent without a known entity
    assert collect.detect("hardware of host99") is None                       # no hashtag of that name


def test_collect_gathers_both_parts_of_a_page_in_order_and_names_what_it_left_out(gd, monkeypatch):
    _page(gd)
    monkeypatch.setattr(vector, "rerank_fn", lambda q, docs: [5.0 if ("RAM" in d or "NVMe" in d) else 0.0 for d in docs])
    hits = g.recall("host7", limit=1)
    ctx, counts = collect.collect("hardware host7", hits, collect.detect("hardware of host7"))
    assert counts["objects_with_entity_tag"] == 3 and counts["pages"] == 2  # the host page (two parts) and the cluster page
    assert ctx.index("| RAM | 128 GB |") < ctx.index("Two NVMe in a ZFS mirror")  # part 1 before part 2, one page
    assert "parts 2 of 2 found" in ctx
    assert "Printer" not in ctx  # another entity's page is never a candidate
    line = collect.note(counts)
    assert "objects carry the entity hashtag" in line and "scored by reranker" in line


def test_a_failing_reranker_never_costs_the_answer(gd, monkeypatch):
    _page(gd)
    monkeypatch.setattr(vector, "rerank_fn", lambda q, docs: (_ for _ in ()).throw(OSError("down")))
    ctx, counts = collect.collect("hardware host7", [], collect.detect("hardware of host7"))
    assert "RAM" in ctx and counts["scored_by"].startswith("word overlap (reranker failed")


def test_vi_recall_switches_by_itself_and_collect_false_switches_it_off(gd, monkeypatch):
    _page(gd)
    monkeypatch.setattr(vector, "rerank_fn", lambda q, docs: [1.0 for _ in docs])
    on = tools._recall({"filter": "Welche Ausstattung hat host7?", "reinforce": False}, "a", None, None)
    off = tools._recall({"filter": "Welche Ausstattung hat host7?", "reinforce": False, "collect": False}, "a", None, None)
    assert "collect mode (entity host:host7.example.org" in on and "| RAM | 128 GB |" in on
    assert "collect mode" not in off


def test_the_budget_holds_and_private_objects_of_others_stay_invisible(gd, monkeypatch):
    _page(gd)
    g.new_object("insight", "host7 secret note", ["host:host7.example.org"], {"content": "secret " * 50}, reason="T",
                 instance="other", visibility="private")
    monkeypatch.setattr(vector, "rerank_fn", lambda q, docs: [1.0 for _ in docs])
    token = g.current_instance.set("a")  # reset afterwards: a leaked instance broke 8 later tests of the suite
    try:
        ctx, counts = collect.collect("host7", [], collect.detect("host7"), min_tokens=10, max_tokens=60)
    finally:
        g.current_instance.reset(token)
    assert counts["tokens"] <= 60 and "secret" not in ctx
