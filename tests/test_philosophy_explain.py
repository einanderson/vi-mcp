"""philosophy-v2: a package stores only its clingo-object seed; prose is a translation checked by the round trip
prose → rebuilt seed → the original's answers and duties (clingo_object/philosophy_explain.py). The models are replaced by functions
here; the measurement with real models is in the CHANGELOG."""
import json
from pathlib import Path

import pytest

pytest.importorskip("clingo")

from clingo_object import philosophy as ph  # noqa: E402
from clingo_object import philosophy_explain as pe  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kollektiv_principles_1.1.0.json"


def _rules(pkg, sep=" & "):
    return "\n".join(r.replace(" & ", sep) for _b, rs in ph.blocks(pkg["seed"])[0] for r in rs)


def test_no_package_stores_prose_and_every_rule_lies_in_a_marked_block():
    for name in ph.BUILTIN:
        pkg = ph.load(name)
        assert "principles" not in pkg and pkg["format"] == "philosophy-v2"
        bl, problems = ph.blocks(pkg["seed"])
        assert not problems and bl
        assert all(ph.MARKER.match(x) for x in pkg["seed"].splitlines() if x.startswith("#"))  # no title, no prose
    assert "principles" not in ph.PRIORITY_PROFILES["maximal-candidates-v1"]


def test_the_historical_prose_is_frozen_outside_the_package_and_never_loaded():
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fx["version"] == "1.1.0" and "HISTORICAL" in fx["_note"] and "NEVER loaded" in fx["_note"]
    assert len(fx["principles"]) == 7 and fx["principles"][0]["id"] == "P1"
    module = (Path(ph.__file__).parent / "philosophy_kollektiv.py").read_text(encoding="utf-8")
    assert fx["principles"][0]["text"] not in module and "kollektiv_principles" not in module


def test_the_hash_never_covers_a_derived_translation():
    pkg = ph.load("kollektiv")
    assert ph.package_hash({**pkg, "translation": {"whole": {"prose": "anything"}}}) == ph.package_hash(pkg)


def test_normalize_fixes_syntax_never_meaning():
    asp = "- counts(S, P, O) :- line(S, P, O, I), author(S, A), I != A.\nabsent(S) <= closed(S), not pos(S)\nnever(value(S, V), ~reasoned(S))"
    assert pe.normalize(asp).splitlines() == ["counts(S, P, O) <= line(S, P, O, I) & author(S, A) & (I != A)",
                                              "absent(S) <= closed(S) & ~pos(S)", "never(value(S, V) & ~reasoned(S))"]
    # measured with qwen27b (15.11.2026): it wrote a constraint as a rule head four rounds in a row
    assert pe.normalize("never(value(S,V)) <= value(S,V) & ~grounded(S)") == "never(value(S,V) & value(S,V) & ~grounded(S))"


def test_a_faithful_back_translation_agrees_everywhere_and_a_lossy_one_does_not():
    pkg = ph.load("kollektiv")
    good = pe.round_trip(pkg, lambda m: "prose", lambda m: _rules(pkg, ", "), random_n=20, per_block=False,
                         names={"translator": "t", "back_translator": "b"})
    assert good["whole"]["agreement"] == 1.0 and good["whole"]["worlds"] == len(pkg["worlds"]) + 20 and good["derived"]
    lossy = _rules(pkg).replace(" & (I != A)", "")  # what the measured back-translator did: self-confirmation dropped
    bad = pe.round_trip(pkg, None, lambda m: lossy, prose="given prose", control="reference prose", random_n=40, per_block=False)
    assert bad["whole"]["agreement"] < 1.0 and bad["control"]["agreement"] == bad["whole"]["agreement"]
    assert any("control (reference prose" in x for x in pe.summary_lines(bad))


def test_each_block_is_measured_on_its_own():
    pkg = ph.load("kollektiv")
    blocks = dict(ph.blocks(pkg["seed"])[0])

    def back(msgs):  # a back-translator that gets every block right except P1, where it drops the guard
        text = msgs[0]["content"]
        b = next(k for k in blocks if f"PROSE-{k}" in text)
        rules = "\n".join(blocks[b])
        return rules.replace(" & (I != A)", "") if b == "P1" else rules

    def trans(msgs):
        text = msgs[-1]["content"]
        for k in blocks:
            if f"rules of block {k} " in text:
                return f"PROSE-{k}"
        return "PROSE-P1"  # the whole seed: measured like P1 here
    r = pe.round_trip(pkg, trans, back, random_n=30)
    assert r["blocks"]["P1"]["agreement"] < 1.0
    assert all(r["blocks"][b]["agreement"] == 1.0 for b in blocks if b != "P1")


def test_the_repair_loop_feeds_clingos_message_back():
    pkg = ph.load("messung")
    answers = iter(["value(S, t) <= ~pos(S)", _rules(pkg)])  # S only in a negated literal: unsafe, clingo refuses
    seen = []

    def back(msgs):
        seen.append(msgs)
        return next(answers)
    r = pe.back_translate(pkg, "prose", back)
    assert r["rounds"] == 2 and r["errors"] and "clingo refused this" in seen[1][-1]["content"]
    assert pe.back_translate(pkg, "prose", lambda m: "value(S <= ", rounds=2).get("failed")


def test_explain_store_and_show_mark_the_translation_as_derived(gd, monkeypatch):
    from vi_mcp import philosophy, tools
    pkg = ph.load("kollektiv")
    monkeypatch.setattr(pe, "chat_client", lambda url, model=None, **k: (lambda m: _rules(pkg)) if "judge" in url else (lambda m: "The rules say …"))
    with pytest.raises(tools.Rejected, match="back-translator"):
        tools.call_tool("vi_philosophy", {"action": "explain", "name": "kollektiv", "url": "http://t/v1"}, "a", False)
    with pytest.raises(tools.Rejected, match="install it first"):
        tools.call_tool("vi_philosophy", {"action": "explain", "name": "kollektiv", "url": "http://t/v1", "judge_url": "http://judge/v1",
                                          "random": 5, "whole_only": True, "store": True}, "a", False)
    tools.call_tool("vi_philosophy", {"action": "install", "name": "kollektiv"}, "a", False)
    out = tools.call_tool("vi_philosophy", {"action": "explain", "name": "kollektiv", "url": "http://t/v1", "model": "m1",
                                            "judge_url": "http://judge/v1", "judge_model": "m2", "random": 5, "store": True}, "a", False)
    assert "DERIVED, never a source" in out and "stored as a derived translation" in out and "block P7" in out
    shown = tools.call_tool("vi_philosophy", {"action": "show", "name": "kollektiv"}, "a", False)
    assert "DERIVED translation (current)" in shown and "translator m1, back-translator m2" in shown and "The rules say" in shown
    t = philosophy.stored_translation("kollektiv")
    changed = {**pkg, "seed": pkg["seed"] + "# P8\nobligatory(report(S)) <= value(S, f)\n"}
    assert "STALE" in philosophy.translation_lines(changed, t)[0]  # a changed seed makes it stale, it never updates itself
    same = philosophy.explain("kollektiv", "http://judge/v1", None, "http://judge/v1", None, random_n=0, per_block=False)
    assert "not independent" in same["warning"]
