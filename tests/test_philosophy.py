"""0.66.0: philosophy packages — equip an AI with a philosophy as a checked clingo-object rule set (the seed)."""
import copy

import pytest

from clingo_object import philosophy as ph


def test_the_collective_package_passes_every_check_and_judges_like_its_principles():
    pkg = ph.load("kollektiv")
    r = ph.check(pkg)
    assert r["ok"], r["errors"]
    assert r["coverage"]["value"] == {"t": True, "f": True, "b": True, "n": True}
    assert all(r["coverage"]["acceptance"].values()) and len(r["worlds"]) >= 12 and r["hash"] == ph.package_hash(pkg)
    v = ph.evaluate(pkg, "line(q, pos, fam_a, x)\nline(q, neg, fam_b, y)\nauthor(q, z)\nclosed(q)")
    assert v["answers"]["value"] == ["value(q,b)"] and "obligatory(report(q))" in v["duties"]
    assert "forbidden(assert_unqualified(q))" in v["duties"] and v["violations"] == []
    own = ph.evaluate(pkg, "line(q, pos, fam_a, z)\nauthor(q, z)\nclosed(q)")  # self-confirmation: unknown, never true
    assert own["answers"]["value"] == ["value(q,n)"]


def test_the_template_is_a_valid_start_and_a_directory_round_trips(tmp_path):
    t = ph.template("probe", "tester")
    assert ph.check(t)["ok"]
    d = ph.write(ph.load("kollektiv"), str(tmp_path / "k"))
    back = ph.load(str(d))
    assert ph.package_hash(back) == ph.package_hash(ph.load("kollektiv")) and ph.check(back)["ok"]
    with pytest.raises(ph.PackageError, match="neither a built-in"):
        ph.load(str(tmp_path / "missing"))


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["vocabulary"].pop("grund"), "grund/3 is used by the seed but not declared"),
    (lambda p: p["vocabulary"]["two"].update(arity=3), "predicate two: used with arity [2], declared 3"),
    (lambda p: p.update(seed=p["seed"] + "neg(S) <= claim(S) & ~pos(S)\n") or p["vocabulary"].update(claim={"arity": 1, "kind": "input"}),
     "default negation in a rule deriving neg"),  # Codex (a): missing support must never become opposition
    (lambda p: p["principles"].pop(), "no principle names it"),
    (lambda p: p["worlds"][0]["holds"].append("value(s1,f)"), "expected value(s1,f), not derived"),
    (lambda p: p["answers"]["value"]["values"].append("x"), "answer value = x is produced by no world"),
    (lambda p: p["worlds"][1].update(facts=p["worlds"][0]["facts"], holds=[], absent=[]), "gives the same answers and duties"),
    (lambda p: p["profile"].update(semantics="anything-goes"), "profile.semantics 'anything-goes' unknown"),
    (lambda p: p.update(seed="value(S, t) <= "), "the seed does not translate"),
])
def test_check_refuses_what_would_let_a_philosophy_lie_or_stay_silent(mutate, message):
    pkg = copy.deepcopy(ph.load("kollektiv"))
    mutate(pkg)
    r = ph.check(pkg)
    assert not r["ok"] and any(message in e for e in r["errors"]), r["errors"]


def test_a_seed_that_calls_everything_unknown_fails_the_sensitivity_check():
    """Codex's review: a seed that answers 'unknown' for everything conforms to itself perfectly — coverage and counterfactuals
    refuse it."""
    pkg = copy.deepcopy(ph.load("kollektiv"))
    pkg["seed"] = "value(S, n) <= closed(S)\n"
    pkg["vocabulary"] = {"closed": {"arity": 1, "kind": "input"}, "value": {"arity": 2, "kind": "answer"}}
    pkg["answers"] = {"value": {"position": 1, "values": ["t", "f", "b", "n"]}}
    pkg["principles"] = [{"id": "P", "text": "Nothing is known.", "defines": ["value"]}]
    for w in pkg["worlds"]:
        w["holds"], w["absent"] = [], []
    r = ph.check(pkg)
    assert not r["ok"]
    assert any("value = t is produced by no world" in e for e in r["errors"])
    assert any("gives the same answers and duties" in e for e in r["errors"])


def test_the_tool_and_install_store_a_checked_philosophy_as_a_rule_set(gd):
    from vi_mcp import tools
    from vi_mcp import memory as g
    assert "Philosophy packages" in tools.call_tool("vi_philosophy", {}, "a", False)
    assert "call vi_philosophy" in tools.RULES
    with pytest.raises(tools.Rejected):
        tools.call_tool("vi_philosophy", {}, "offen", False)  # needs a token
    assert "passes every check" in tools.call_tool("vi_philosophy", {"action": "check", "name": "kollektiv"}, "a", False)
    assert "P1 No self-confirmation" in tools.call_tool("vi_philosophy", {"action": "show", "name": "kollektiv"}, "a", False)
    out = tools.call_tool("vi_philosophy", {"action": "evaluate", "name": "kollektiv",
                                            "facts": "line(q, pos, fam_a, x)\nauthor(q, z)\nclosed(q)"}, "a", False)
    assert "acceptance(q,tentative)" in out and "forbidden(assert_unqualified(q))" in out
    with pytest.raises(tools.Rejected, match="needs name"):
        tools.call_tool("vi_philosophy", {"action": "check"}, "a", False)
    msg = tools.call_tool("vi_philosophy", {"action": "install", "name": "kollektiv"}, "a", False)
    oid = msg.split()[1]
    o = g.load(oid)
    assert o.kind == "regel" and "philosophie:kollektiv" in o.hashtags and o.visibility == "intern"
    assert o.data["philosophie"]["hash"] == ph.package_hash(ph.load("kollektiv")) and o.data["logik"]["quelle"] == ph.load("kollektiv")["seed"]
    assert "kollektiv 1.0.0" in tools.call_tool("vi_philosophy", {"action": "list"}, "a", False)


def test_install_refuses_a_package_that_fails_its_checks(gd, tmp_path):
    from vi_mcp import philosophy
    pkg = ph.load("kollektiv")
    pkg["worlds"][0]["holds"].append("value(s1,f)")
    d = ph.write(pkg, str(tmp_path / "broken"))
    with pytest.raises(ValueError, match="package refused"):
        philosophy.install(str(d), "a")


def test_the_published_guide_is_generated_from_the_code():
    from pathlib import Path
    doc = Path(__file__).resolve().parent.parent / "docs" / "PHILOSOPHY.md"
    assert doc.read_text(encoding="utf-8").strip() == ph.GUIDE.strip(), "regenerate: vi-mcp philosophy guide > docs/PHILOSOPHY.md"
