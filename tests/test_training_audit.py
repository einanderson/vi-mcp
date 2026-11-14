"""Regression cases measured during the VI review; no live services or company data."""
import hashlib
import json
from pathlib import Path

import pytest

from vi_mcp import dataset, evaluation, finetune, incubator, learning, metrics, tools, training_eval
from vi_mcp import memory as g


def test_source_page_and_duplicate_content_cannot_leak(gd, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path / "eval")
    target = g.new_object("erkenntnis", "evaluation chunk", [], {"quelle": "https://docs.example/page", "inhalt": "unique reference body"})
    g.new_object("erkenntnis", "another chunk of same page", [], {"quelle": "https://docs.example/page · Version 2", "inhalt": "other text"})
    g.new_object("erkenntnis", "duplicate on a different page", [], {"quelle": "https://docs.example/duplicate", "inhalt": "unique reference body"})
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval/recall.json").write_text(json.dumps([{"target": target.id, "group": evaluation.group_of(target), "frage": "question"}]))
    built = dataset.build(agentic=False)
    assert not built["qa"] and built["held_out"] == 1
    assert built["skipped"]["evaluation source page or duplicate content (held out)"] == 2
    (tmp_path / "eval/recall.json").write_text("broken")
    with pytest.raises(ValueError, match="holdout"):
        dataset.build(agentic=False)


def test_all_training_stages_share_source_and_object_split(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    qa = [{"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "answer"}],
           "meta": {"id": str(i), "group": "page-" + str(i // 2)}} for i in range(30)]
    for name, rows in (("qa", qa), ("logic", [{"meta": r["meta"], "messages": []} for r in qa]),
                       ("agentic", [{"meta": r["meta"], "messages": []} for r in qa])):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = finetune._split(d / "qa.jsonl", tmp_path / "split")
    def rows(name):
        return [json.loads(s) for s in (tmp_path / "split" / name).read_text().splitlines()]
    held = {r["meta"]["id"] for r in rows("test.jsonl")}
    assert held and not held & {r["meta"]["id"] for r in rows("logic.train.jsonl")}
    assert not held & {r["meta"]["id"] for r in rows("agentic.train.jsonl")}
    assert not {r["meta"]["group"] for r in rows("train.jsonl")} & {r["meta"]["group"] for r in rows("test.jsonl")}
    assert result["groups"] == 15


def test_noninferiority_requires_excluding_material_harm(gd, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path / "eval")
    def criterion(d, se):
        monkeypatch.setattr(metrics, "holdback_effect", lambda *_: {"zustand": "gemessen", "differenz": d, "standardfehler": se, "urteil": "fixture"})
        return next(c for c in incubator.status()["kriterien"] if c["kriterium"] == "hints_no_harm")
    assert criterion(-0.15, 0.10)["zustand"] == "offen"
    assert criterion(0, 0.10)["zustand"] == "offen"
    assert criterion(0.01, 0.005)["zustand"] == "erfuellt"


def test_holdout_and_validation_do_not_divide_one_action():
    ids = [f"action-{i // 4}" for i in range(412)]
    n = learning._holdout_size([i % 2 for i in range(412)], ids)
    assert not set(ids[:-n]) & set(ids[-n:])
    cut = learning._group_cut(ids, 330)
    assert not set(ids[:cut]) & set(ids[cut:])


def test_retained_champion_keeps_its_metrics_and_date(gd):
    learning._save({"trainiert": "2026-01-01", "auc_holdout": .91, "n": 1000, "zustand": "aktiv"}, {"tool:demo": 1.0}, 0, "aktiv", True)
    learning._keep_champion({"trainiert": "2026-02-01", "auc_holdout": .61, "n": 1200, "champion": {"zustand": "schlechter", "urteil": "fixture"}})
    m = learning.model()
    assert m.data["auc_holdout"] == .91 and m.data["trainiert"] == "2026-01-01"
    assert m.data["herausforderer"]["auc_holdout"] == .61


def test_solver_success_and_one_positive_atom_do_not_verify_program(gd, tmp_path, monkeypatch):
    pytest.importorskip("clingo")
    from vi_mcp import logic
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path / "eval")
    g.new_object("erkenntnis", "DMZ", ["host:gate1", "zone:dmz"], {"quelle": "fixture"})
    g.new_object("erkenntnis", "internal", ["host:office1", "zone:internal"], {"quelle": "fixture"})
    wrong = 'ans(H) <= (O in objects("host:*")) & (O.host == H)'
    monkeypatch.setattr(logic, "agent_llm", lambda _: wrong)
    tools.call_tool("vi_logic", {"action": "answer", "question": "Which hosts are in the DMZ?"}, "a", False)
    tools.call_tool("vi_logic", {"action": "rate", "source": wrong, "conclusion": 'ans("gate1")', "correct": True}, "a", False)
    assert not any(r["meta"]["kind"] == "asked" for r in dataset.build()["logic"])
    v = json.loads(tools.call_tool("vi_logic", {"action": "verify", "source": wrong, "question": "Which hosts are in the DMZ?",
        "expected_answers": ['ans("gate1")'], "reference": "independent fixture inventory"}, "a", False))
    assert v["state"] == "refuted" and v["unexpected"] == 1


@pytest.mark.parametrize("bad", ["ans(X)", 'port("gate1")'])
def test_a_broken_reference_is_refused_not_booked_as_a_wrong_program(gd, bad):
    """Review 14.11.2026 (Claude): a variable or another predicate in the reference used to be accepted and rated the program wrong —
    in the machine that became a KTO negative. One shared rule (notation.reference_answers) now refuses it in verify and in rate."""
    pytest.importorskip("clingo")
    g.new_object("erkenntnis", "DMZ", ["host:gate1", "zone:dmz"], {"quelle": "fixture"})
    src = 'ans(H) <= (O in objects("zone:dmz")) & (O.host == H)'
    with pytest.raises(tools.Rejected, match="ground|predicate"):
        tools.call_tool("vi_logic", {"action": "verify", "source": src, "question": "Which hosts are in the DMZ?",
                                     "expected_answers": ['ans("gate1")', bad], "reference": "independent fixture"}, "a", False)
    call = lambda action, **kw: json.loads(tools.call_tool("vi_machine", {"action": action, "name": "m", **kw}, "a", False))
    call("create")
    call("execute", instruction="tell.fact gate1", payload={"text": "gate", "hashtags": ["zone:dmz", "host:gate1"], "source": "fixture"})
    ex = call("execute", instruction="ask.rule", payload={"source": 'ans(H) <= (O in objects("zone:dmz")) & (O.host == H)'})
    with pytest.raises(tools.Rejected, match="ground|predicate"):
        call("execute", instruction="rate " + ex["receipt"], payload={"expected_answers": [bad], "reference": "fixture", "question": "q"})
    assert call("status")["checks"] == 0
    assert not training_eval.evaluate_program(src, {"expected": [bad]})["correct"]


def test_program_evaluation_checks_semantics_and_extra_answers():
    pytest.importorskip("clingo")
    case = {"expected": ['ans("safe")']}
    assert training_eval.evaluate_program('ans("safe")', case)["correct"]
    r = training_eval.evaluate_program('ans("safe")\nans("unsafe")', case)
    assert not r["correct"] and r["unexpected"] == 1 and r["recall"] == 1
    assert not training_eval.evaluate_program("import os", case)["correct"]
    assert training_eval.evaluate_program('other("x")', {"expected": []})["correct"]
    assert not training_eval.evaluate_program('ans("safe") <= ~other\nother <= ~ans("safe")', case)["correct"]


def test_correctness_needs_independent_judgment_and_uncertainty():
    with pytest.raises(ValueError, match="independent reference"):
        training_eval.summarize_judgments([{"base_correct": False, "tuned_correct": True}])
    one = training_eval.summarize_judgments([{"base_correct": False, "tuned_correct": True, "reference": "fixture"}])
    assert one["lower_95"] < 0  # no 100% certainty from a single success
    many = training_eval.summarize_judgments([{"id": str(i), "base_correct": False, "tuned_correct": True, "reference": "fixture"} for i in range(100)])
    assert many["lower_95"] > 0


def test_champion_does_not_count_repeated_hints_as_independent_actions(gd):
    champ = {"gewichte": {"tool:demo": 2.0}, "trainiert": "2000-01-01"}
    action = g.new_object("erfahrung", "one action", [], {})
    repeated = [({"tool:demo": 1.0}, 1, action.id)] * learning.CHAMPION_MIN_N
    result = learning.compare_champion(champ, {"tool:demo": -2.0}, 0, repeated)
    assert result["zustand"] == "nicht_bestimmbar" and result["n"] == 1
