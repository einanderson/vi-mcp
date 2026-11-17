"""17.11.2026: vi_feedback answered "not shown … in the last 60 min" for every hint shown for a user prompt (vi_prompt_context)
and after a failure (error hints) — on a fresh install the only hints shown at all (0 of 273 actions there had a hint shown before them) —
although the hook's Stop request asks for exactly those. Now they are booked on the latest action and count for the bandit."""
from vi_mcp import bandit, intuition, session, tools
from vi_mcp import memory as g


def _k(text, tags):
    return g.new_object("insight", text, tags, {"source": "test 17.11.2026"}, instance="b")


def test_feedback_on_a_prompt_hint_is_booked_on_the_latest_action(gd, monkeypatch):
    monkeypatch.setattr(intuition, "_prompt_shown", {}, raising=False)
    k = _k("web1 does not boot again by itself after a power cut", ["host:web1.example", "topic:power"])
    s = session.begin("a", "t")
    with session.on_behalf(s):
        ids = intuition.prompt_hints("why does web1 not boot after the power cut", "a")["ids"]
        assert k.id in ids
        v = intuition.before("Bash", {"command": "ping -c1 web1.example"}, "a", False)
        intuition.after("Bash", {"command": "ping -c1 web1.example"}, True, 5, None, None, v["vorher_id"], "a", False)
    out = tools.call_tool("vi_feedback", {"helped": [k.id], "wrong": ["erkenntnis:nie-gezeigt"]}, "a", False)
    assert "feedback recorded for 1 hint(s)" in out and "erkenntnis:nie-gezeigt" in out
    e = [o for _s, o, _g in g.recall("kind:experience", viewer=None)][0]
    assert e.data["feedback"] == {k.id: 1} and e.data["prompt_shown"] == [k.id]
    obs = list(bandit._observations())
    assert any(k.id in shown for _t, _ok, shown, _per in obs)  # counts for the bandit, with its explicit label
    per = next(per for _t, _ok, shown, per in obs if k.id in shown)
    assert per[k.id][0] == 1


def test_a_prompt_hint_of_another_instance_is_not_booked(gd, monkeypatch):
    monkeypatch.setattr(intuition, "_prompt_shown", {}, raising=False)
    k = _k("web1 does not boot again by itself after a power cut", ["host:web1.example", "topic:power"])
    intuition.prompt_hints("why does web1 not boot after the power cut", "other")
    s = session.begin("a", "t")
    with session.on_behalf(s):
        v = intuition.before("Bash", {"command": "true"}, "a", False)
        intuition.after("Bash", {"command": "true"}, True, 5, None, None, v["vorher_id"], "a", False)
    assert "feedback recorded for 0 hint(s)" in tools.call_tool("vi_feedback", {"wrong": [k.id]}, "a", False)


def test_unrated_prompt_hints_do_not_enter_the_bandit(gd, monkeypatch):
    monkeypatch.setattr(intuition, "_prompt_shown", {}, raising=False)
    g.new_object("experience", "Bash run", ["tool:Bash", "ok:ja"],
                 {"before": {"found": [], "shown": []}, "prompt_shown": ["erkenntnis:x"], "error_hints": ["erkenntnis:y"]}, instance="a")
    assert all(not shown for _t, _ok, shown, _per in bandit._observations())


def test_feedback_on_an_error_hint_is_booked(gd):
    k = _k("x", ["tool:Bash"])
    e = g.new_object("experience", "Bash run", ["tool:Bash", "ok:nein"],
                     {"before": {"found": [], "shown": []}, "error_hints": [k.id], "error_held_back": False}, instance="a")
    assert "feedback recorded for 1 hint(s)" in tools.call_tool("vi_feedback", {"wrong": [k.id]}, "a", False)
    assert g.load(e.id).data["feedback"] == {k.id: -1}
