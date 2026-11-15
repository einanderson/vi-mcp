"""The provenance adapter (Unreleased): what the memory stores about a knowledge object becomes kollektiv 1.3.0's input facts — origins,
copies, model-made origins, the weight of evidence, camps — and the seed's acceptance is compared with today's mode, writing nothing."""
import hashlib
import re

import pytest

pytest.importorskip("clingo")

from clingo_object import philosophy as ph  # noqa: E402
from vi_mcp import memory as g  # noqa: E402
from vi_mcp import philosophy_provenance as pv  # noqa: E402


def _obj(text="Port 8080 serves the admin UI", quelle="https://docs.example.org/admin", instance="a", tags=("thema:t",)):
    return g.new_object("insight", text, list(tags), {"quelle": quelle}, reason="T", instance=instance)


def _acceptance(o):
    facts, info = pv.facts_for(g.load(o.id))
    atoms = pv._solve(ph.load("kollektiv"), facts)
    acc = sorted(m.group(1) for a in atoms if (m := re.fullmatch(rf'acceptance\("{re.escape(o.id)}",(\w+)\)', a)))
    held = sorted(m.group(1) for a in atoms if (m := re.fullmatch(rf'grund\("{re.escape(o.id)}",held_back,(\w+)\)', a)))
    return acc, held, facts, info


def _entries(o, n, source, family=lambda k: f"fam{k}"):
    o = g.load(o.id)
    o.data["bestaetigungen"] = [{"instanz": f"i{k}", "familie": family(k), "quelle": source(k), "zeit": g.now()} for k in range(n)]
    o.confirmed = n
    return g.save(o)


def test_1000_confirmations_copied_from_one_origin_stay_tentative(gd):
    """P8: a thousand confirmers citing the same post are one origin. Today's count of independent lines (one per model family) calls
    it confirmed by a thousand lines; the seed holds it tentative."""
    o = _entries(_obj(), 1000, lambda k: "https://news.example/post-1")
    assert pv.today(o) == "confirmed_2plus" and o.foreign_confirmations() == 1000
    acc, _held, facts, _info = _acceptance(o)
    assert acc == ["tentative"] and f'closed("{o.id}")' in facts
    two = _entries(_obj("Port 9090 serves the metrics"), 1000, lambda k: f"https://src{k % 2}.example/report")  # two real origins
    assert _acceptance(two)[0] == ["accepted"]


def test_a_record_cut_at_its_cap_concludes_nothing_and_invents_no_origin(gd):
    o = _obj()
    for k in range(150):
        o = g.load(o.id).confirm(f"ki@i{k}", "seen", instance=f"i{k}", family=f"fam{k}", source="https://news.example/post-1")
    o = g.load(o.id)
    assert len(o.data["bestaetigungen"]) == 100 and o.confirmed == 150
    acc, _held, facts, info = _acceptance(o)
    assert acc == [] and not any(f.startswith("closed(") for f in facts) and info["not_closed_record"] == 1
    assert sum(f.startswith("line(") for f in facts) == 100  # the 50 dropped entries are not made up as origins


def test_a_withdrawal_by_the_author_is_not_a_refutation_but_two_foreign_families_reject(gd):
    own = _obj()
    own = g.load(own.id).refute("ki@a", "my mistake", instance="a", family="claude")
    acc, _held, facts, _info = _acceptance(own)
    assert pv.today(own) == "withdrawn" and acc == ["withdrawn"] and f'withdrawn("{own.id}")' in facts  # P13, never rejected
    kept = _entries(_obj("Port 6060 serves the API"), 2, lambda k: f"https://src{k}.example/a")
    kept = g.load(kept.id).refute("ki@a", "withdrawn", instance="a", family="claude")
    assert _acceptance(kept)[0] == ["accepted"]  # others' evidence survives the author's withdrawal
    o = _obj("Port 7070 serves nothing")
    o = g.load(o.id).refute("ki@x", "wrong", instance="x", family="gpt")
    assert pv.today(g.load(o.id)) == "disputed" and _acceptance(o)[0] == ["tentative"]
    o = g.load(o.id).refute("ki@y", "wrong too", instance="y", family="qwen")
    assert pv.today(o) == "refuted" and _acceptance(o)[0] == ["rejected"]


def test_model_made_and_copied_origins_add_nothing(gd):
    base = _obj("Backups run at 02:00", quelle="https://ops.example/backup")
    llm = _obj("Backups run nightly", quelle=f"derived from {base.id}", instance=g.SERVER_INSTANCE, tags=("herkunft:schlaf-llm",))
    o = _obj("The nightly backup starts at 02:00", quelle="https://ops.example/runbook")
    o = g.load(o.id).confirm("ki@vi", "sleep", instance=g.SERVER_INSTANCE, family="vi")  # automatic, no source: model-made
    o = g.load(o.id).confirm("ki@x", "agrees", instance="x", family="gpt", source=llm.id)  # cites a model-made object
    acc, _held, facts, info = _acceptance(o)
    assert f'machine_made("i:{g.SERVER_INSTANCE}")' in facts and f'machine_made("obj:{llm.id}")' in facts
    assert f'derives_from("obj:{llm.id}", "obj:{base.id}")' in facts and info["derives_from"] == 1
    assert f'evidence_kind("obj:{llm.id}", hearsay)' in facts  # inferred -> hearsay (documented in the module)
    assert acc == ["tentative"]  # only the copied origin's root counts: one origin


def test_hearsay_alone_is_held_back_with_its_reason(gd):
    p1 = _obj("Rumour A", quelle="", instance="p2p-n1", tags=("herkunft:p2p",))
    p2 = _obj("Rumour B", quelle="", instance="p2p-n2", tags=("herkunft:p2p",))
    o = _obj("The datacenter moves in May", quelle="")
    o = g.load(o.id).confirm("ki@x", "heard", instance="x", family="gpt", source=p1.id)
    o = g.load(o.id).confirm("ki@y", "heard", instance="y", family="qwen", source=p2.id)
    acc, held, facts, _info = _acceptance(o)
    assert f'evidence_kind("obj:{p1.id}", hearsay)' in facts and acc == ["tentative"] and held == ["hearsay_only"]
    assert f'foreign("{p1.id}")' in pv.facts_for(g.load(p1.id))[0]


def test_every_known_camp_is_given_and_the_seed_names_a_partly_known_one(gd):
    o = _obj()
    o = g.load(o.id)
    o.data["bestaetigungen"] = [{"instanz": "x", "familie": "gpt", "quelle": "https://a.example/1", "zeit": g.now()},
                                {"instanz": "y", "quelle": "https://b.example/2", "zeit": g.now()}]
    o.confirmed = 2
    o = g.save(o)
    acc, held, facts, info = _acceptance(o)
    assert facts.count(f'camp("q:https://a.example/1", "gpt")') == 1 and info["camps"] == 1  # every known camp is given
    assert acc == ["tentative"] and held == ["camp_unknown"]  # the seed names the partly known camp, never one_camp
    o.data["bestaetigungen"][1]["familie"] = "gpt"  # both known, the same camp: P10 holds it back
    o = g.save(o)
    acc, held, _facts, _info = _acceptance(o)
    assert acc == ["tentative"] and held == ["one_camp"]


def test_untestable_and_extraordinary_are_never_guessed(gd):
    facts, _info = pv.facts_for(g.load(_obj("Everything happens for a reason").id))
    assert not any(f.startswith(("untestable(", "testable_by(", "extraordinary(", "verified(")) for f in facts)
    assert any(f.startswith("source(") for f in facts)  # the own URL counts nothing until verified (P7)


def test_shadow_compares_and_changes_nothing(gd, capsys):
    a = _entries(_obj(), 300, lambda k: "https://news.example/post-1")
    b = _obj("Port 7070 serves nothing")
    g.load(b.id).refute("ki@a", "mine, wrong", instance="a", family="claude")
    _obj("Something unconfirmed")

    def digest():
        with g.lock:
            return hashlib.sha256(repr(g.db().execute("SELECT id, mode, data, confirmed, refuted, last_at FROM object ORDER BY id").fetchall())
                                  .encode()).hexdigest()
    before = digest()
    r = pv.shadow()
    assert digest() == before  # shadow mode: no object changed
    assert r["objects"] == 3 and r["pairs"]["confirmed_2plus -> tentative"] == 1 and r["pairs"]["withdrawn -> withdrawn"] == 1  # 0.67.0: memory names the withdrawal itself
    assert r["pairs"]["unconfirmed -> unresolved"] == 1 and r["agree"] == 2 and r["disagree"] == 1
    assert a.id in r["examples"]["confirmed_2plus -> tentative"]
    assert "never filled" in "\n".join(pv.shadow_lines(r))
    with pytest.raises(SystemExit, match="kollektiv only"):
        pv.main_shadow("messung", None, None, None)
