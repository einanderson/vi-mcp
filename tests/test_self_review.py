"""0.69.3: an AI may check its own work (the user, 17.11.2026: "in our environment there is only Claude"). Each instance is its own
line, the author's own instance counts half and is marked self, the server never; a self-grade of a conversation is an agent grade,
never a human one, and counts for learning only once calibrated."""
import pytest

from vi_mcp import memory as g, scorer, session, tools
from vi_mcp.tools import Rejected


def _fact(instance="a"):
    return g.new_object("insight", "service y needs 2 GB RAM", ["topic:y"], {"source": "free -m on web1, 10.11. 14:00"}, reason="T",
                        instance=instance)


def test_self_review_counts_half_and_is_marked(gd):
    o = _fact()
    g.load(o.id).confirm("ki@a", "measured again on web1 with free -m", "a", family="claude", source="free -m on web1, 17.11. 09:00")
    o = g.load(o.id)
    assert o.self_reviewed() and o.foreign_confirmations() == 0 and o.confirmation_weight() == g.SELF_WEIGHT
    assert f"self-review, weight {g.SELF_WEIGHT}" in o.changes[-1]["what"]
    g.load(o.id).confirm("ki@b", "same on web2", "b", family="claude", source="free -m on web2")
    o = g.load(o.id)
    assert o.foreign_confirmations() == 1 and o.confirmation_weight() == 1 + g.SELF_WEIGHT


def test_instances_of_one_family_are_separate_lines_but_the_server_never_counts(gd):
    o = _fact()
    for i in ("b", "c"):
        g.load(o.id).confirm(f"ki@{i}", "checked", i, family="claude", source=f"free -m on {i}")
    g.load(o.id).confirm("vi", "sleep", g.SERVER_INSTANCE, source="sleep")
    o = g.load(o.id)
    assert o.independent_lines() == {"b", "c"} and o.confirmation_weight() == 2.0


def test_self_confirmation_from_the_own_source_counts_nothing(gd):
    o = _fact()
    g.load(o.id).confirm("ki@a", "it says so", "a", source="Free -m on web1, 10.11. 14:00.")
    o = g.load(o.id)
    assert not o.self_reviewed() and o.confirmation_weight() == 0


def test_search_bonus_follows_the_weight(gd):
    a, b = _fact(), g.new_object("insight", "service y needs 2 GB RAM too", ["topic:y"], {"source": "x"}, instance="a")
    g.load(a.id).confirm("ki@a", "re-measured", "a", source="free -m on web1 again")
    rated = {o.id: d for _s, o, d in g.recall("service y needs RAM", limit=10)}
    assert rated[a.id]["wichtigkeit"] > rated[b.id]["wichtigkeit"]


def test_two_claude_instances_refute_and_the_author_withdraws(gd):
    o = _fact()
    g.load(o.id).refute("ki@b", "4 GB on web1", "b", family="claude")
    assert g.load(o.id).mode == "disputed"
    g.load(o.id).refute("ki@c", "4 GB measured", "c", family="claude")
    assert g.load(o.id).mode == "refuted"
    p = _fact()
    g.load(p.id).refute("ki@a", "my measurement was wrong", "a")
    assert g.load(p.id).mode == "withdrawn"


def test_vi_rate_stores_an_agent_self_grade_that_counts_only_when_calibrated(gd):
    s = session.begin("a", "claude-code")
    with session.on_behalf(s):
        with pytest.raises(Rejected, match="reason needed"):
            tools.call_tool("vi_rate", {"grade": 0.9, "reason": "fine"}, "a", False)
        with pytest.raises(Rejected, match="between 0 and 1"):
            tools.call_tool("vi_rate", {"grade": 3, "reason": "all done, tests green in turn 12"}, "a", False)
        out = tools.call_tool("vi_rate", {"grade": 0.2, "reason": "the fix in turn 4 broke the import, not repaired", "evidence": [4, 7],
                                          "certainty": "high"}, "a", False)
    assert f"rated {s.serial}: 0.20 as a self-review" in out
    r = next(o for _x, o, _g in g.recall("kind:rating scorer:agent", limit=5, viewer=None))
    assert r.data["review"] == "self" and r.data["evidence"] == [4, 7] and "review:self" in r.hashtags and "scorer:human" not in r.hashtags
    det = scorer.grade_details()[s.serial]
    assert det["quelle"] == "agent" and det["gewicht"] == round(1.0 * 1.0 * scorer.UNCALIBRATED * g.SELF_WEIGHT, 3)
    assert s.serial not in scorer.conversation_grades()  # not calibrated → not a learning grade
    scorer.human(s.serial, "a", 0.9)
    assert scorer.grade_details()[s.serial]["quelle"] == "human"  # a human grade wins
    assert "Agent self-reviews" in tools.call_tool("vi_scorer", {}, "a", False)


def test_vi_rate_only_for_the_own_instance(gd):
    s = session.begin("b", "claude-code")
    with pytest.raises(Rejected, match="belongs to instance b"):
        tools.call_tool("vi_rate", {"grade": 0.5, "reason": "looked at the other agent's work", "serial": s.serial}, "a", False)
