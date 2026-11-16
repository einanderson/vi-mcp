"""search-top4 (15.11.2026): search at 500,000 objects — the full-text pool first, then visibility; results unchanged."""
from vi_mcp import memory as g


def test_the_full_text_pool_gives_the_same_hits_as_the_exact_query_also_when_invisible_rows_fill_it(gd, monkeypatch):
    for i in range(12):  # private objects of another instance match best (repeated word) — invisible to the viewer
        g.new_object("insight", f"postfix postfix postfix relay note {i}", ["topic:mail"], {"content": "postfix relay"}, visibility="private", instance="other")
    for i in range(8):
        g.new_object("insight", f"postfix relay on mail{i}.example", ["topic:mail" if i % 2 else "topic:web"], {"content": "relay"},
                     visibility="shared", instance="a")
    def run():
        return [o.id for _s, o, _w in g.recall("postfix relay", limit=10, viewer="a", or_fallback=True)]
    def run_tag():
        return [o.id for _s, o, _w in g.recall("topic:mail postfix", limit=10, viewer="a")]
    monkeypatch.setattr(g, "FTS_POOL", 100000)  # the pool holds everything: same as the exact query
    exact, exact_tag = run(), run_tag()
    monkeypatch.setattr(g, "FTS_POOL", 5)  # the pool is filled by invisible rows: the exact query must run
    assert run() == exact and run_tag() == exact_tag
    monkeypatch.setattr(g, "FTS_POOL", 100000)
    monkeypatch.setattr(g, "VISIBLE_CHUNK", 2)  # the visibility check in chunks gives the same hits
    assert run() == exact and run_tag() == exact_tag
    assert len(exact) == 8 and len(exact_tag) == 4 and not any(g.load(i).visibility == "private" for i in exact)
