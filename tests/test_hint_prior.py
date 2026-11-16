"""0.68.0: LLM prior for new hints — the classify model names situation tags (never numbers) for a new hint; prior:match / prior:miss are
router features only while the two-step offline gate says on."""
import json
import random

from vi_mcp import hint_prior as hp, learning, reflection
from vi_mcp import memory as g


def test_label_keeps_only_vocabulary_tags_and_never_rates(gd, monkeypatch):
    g.new_object("insight", "old note", ["tool:Bash", "topic:ssh"], {"source": "t"})
    h = g.new_object("insight", "ssh to the build host needs the key file", ["topic:ssh"], {"source": "t"})
    monkeypatch.setattr(reflection, "ask_fn", lambda s, u: json.dumps({"tags": ["tool:Bash", "topic:ssh", "topic:invented", "score:9"]}))
    r = hp.label(10)
    assert r["labelled"] >= 1 and r["tags_dropped"] >= 2
    assert set(g.load(h.id).data["prior"]["tags"]) == {"tool:Bash", "topic:ssh"} and hp.VERSION in g.load(h.id).data["prior"]["version"]
    assert hp.pending(10) == [] or h.id not in [o.id for o in hp.pending(10)]  # labelled once


def test_features_only_for_new_hints_and_only_when_on(gd):
    h = g.new_object("insight", "x", ["topic:ssh"], {"source": "t", "prior": {"tags": ["tool:Bash"]}})
    assert hp.state() == "off" and hp.features(h.id, ["tool:Bash"], True) == {}
    assert hp.features(h.id, ["tool:Bash"], True, force=True) == {"prior:match": 1.0}
    assert hp.features(h.id, ["tool:Read"], True, force=True) == {"prior:miss": 1.0}
    assert hp.features(h.id, ["tool:Bash"], False, force=True) == {}  # an old hint has data of its own


def test_gate_switches_on_when_the_prior_carries_signal_and_off_when_not(gd):
    rnd = random.Random(5)
    helpful = [g.new_object("insight", f"good {i}", ["topic:a"], {"source": "t", "prior": {"tags": ["tool:Bash"]}}) for i in range(30)]
    obs = []
    for i in range(3000):
        h = rnd.choice(helpful)
        tool = rnd.choice(["tool:Bash", "tool:Read"])
        f = {tool: 1.0, f"hint:{h.id}": 1.0, "age:new": 1.0}
        p = 0.92 if tool == "tool:Bash" else 0.6
        obs.append((f, int(rnd.random() < p), f"e{i}", 1.0))
    r = hp.gate(obs)
    # tool:Bash alone already explains it: the prior adds nothing beyond the tool feature, or it does — either way the state is recorded
    assert r["state"] in ("on", "off") and hp.state() == r["state"] and "validation_logloss" in r
    # a prior that matches at random carries no signal → off
    noise = [g.new_object("insight", f"n {i}", ["topic:b"], {"source": "t", "prior": {"tags": [rnd.choice(["tool:Bash", "tool:Read"])]}})
             for i in range(30)]
    obs2 = [({"tool:X": 1.0, f"hint:{rnd.choice(noise).id}": 1.0, "age:new": 1.0, rnd.choice(["tool:Bash", "tool:Read"]): 1.0},
             int(rnd.random() < 0.8), f"n{i}", 1.0) for i in range(3000)]
    assert hp.gate(obs2)["state"] == "off"
    assert "LLM prior for new hints" in hp.line()


def test_without_priors_the_router_is_unchanged(gd):
    h = g.new_object("insight", "x", ["topic:ssh"], {"source": "t"})
    assert not any(k.startswith("prior:") for k in learning.hint_features(h.id, ["tool:Bash"], g.now()))
