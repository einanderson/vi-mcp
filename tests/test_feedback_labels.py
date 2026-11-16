"""0.68.0: more labels on shown hints — measured on production (vi_stats, 7 days to 16.11.2026): explicit feedback for 374 of 13,679 shown
hints (3 %), 1 % adopted. Block 1: the agent names the hints it used in its own vi_after (no extra call); a used hint becomes direct
feedback (+1 after success, −1 after failure), an unnamed one is recorded as declared unused and is NOT a label."""
from vi_mcp import intuition, scorer, session, tools
from vi_mcp import memory as g


def _k(text, tags):
    return g.new_object("insight", text, tags, {"source": "test 16.11.2026"}, instance="b")


def _action(tool_args, ok, used, inst="a", conv="t"):
    s = session.begin(inst, conv)
    with session.on_behalf(s):
        v = intuition.before("Bash", tool_args, inst, False)
        r = json_after = tools.call_tool("vi_after", {"tool": "Bash", "args": tool_args, "ok": ok, "before_id": v["vorher_id"],
                                                       **({"used": used} if used is not ... else {})}, inst, False)
    import json
    return v, json.loads(json_after), r


def _experience():
    return [o for _s, o, _g in g.recall("kind:experience", viewer=None)][0]


def test_used_hints_become_feedback_and_unnamed_ones_are_declared_unused(gd):
    a = _k("uptime on the build host lies after a suspend", ["tool:Bash", "topic:uptime"])
    b = _k("uptime -s shows the boot time on the build host", ["tool:Bash", "topic:uptime"])
    v, out, _ = _action({"command": "uptime"}, True, [a.id, "insight:never-shown"])
    assert set(v["zugriff"]["shown"]) == {a.id, b.id}
    assert out["used"] == {"booked": [a.id], "unused": [b.id], "ignored": ["insight:never-shown"]}
    e = _experience()
    assert e.data["feedback"] == {a.id: 1}
    assert e.data["declared"]["unused"] == [b.id]
    # the label: a used hint after success is a positive direct label; the unused one keeps the action's own outcome label
    assert scorer.label(e.data, True, None, {}, hint=a.id)[2] == "rueckmeldung"
    assert scorer.label(e.data, True, None, {}, hint=b.id)[2] == "ok"


def test_used_before_a_failure_is_negative_and_explicit_feedback_is_not_overwritten(gd):
    a = _k("uptime on the build host lies after a suspend", ["tool:Bash", "topic:uptime"])
    _v, out, _ = _action({"command": "uptime"}, False, [a.id])
    e = _experience()
    assert e.data["feedback"] == {a.id: -1} and scorer.label(e.data, False, None, {}, hint=a.id)[:2] == (0, scorer.WEIGHT["rueckmeldung"])
    # vi_feedback given before the declaration wins
    g.update_data(e.id, lambda d: d.__setitem__("feedback", {a.id: 1}))
    intuition.declare_used(g.load(e.id), [a.id], False)
    assert g.load(e.id).data["feedback"] == {a.id: 1}


def test_none_used_and_not_said_are_different(gd):
    a = _k("uptime on the build host lies after a suspend", ["tool:Bash", "topic:uptime"])
    _v, out, _ = _action({"command": "uptime"}, True, "none")
    assert out["used"] == {"booked": [], "unused": [a.id], "ignored": []}
    assert "feedback" not in _experience().data or _experience().data["feedback"] == {}
    g.db().execute("DELETE FROM object WHERE kind = 'experience'")
    _v, out, _ = _action({"command": "uptime"}, True, ...)  # an older client: no used argument at all
    assert "used" not in out and "declared" not in _experience().data


def test_used_must_be_a_list_or_a_string(gd):
    import pytest
    with pytest.raises(tools.Rejected):
        tools._used_arg({"a": 1})
    assert tools._used_arg("insight:a, insight:b") == ["insight:a", "insight:b"] and tools._used_arg([]) == []
