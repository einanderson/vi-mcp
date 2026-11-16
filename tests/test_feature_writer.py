"""0.68.0: the feature writer — the language model reads the router's weights and errors and writes feature RULES (no numbers, no code);
the logistic regression fits their weights; a rule set is kept only if the paired holdout log loss improves by > 2 SE."""
import json
import random

import pytest

from vi_mcp import feature_writer as fw, learning, reflection
from vi_mcp import memory as g


def test_check_accepts_the_rule_language_and_rejects_everything_else():
    ok, why = fw.check({"rules": [
        {"name": "ssh_timeout_with_key_hint", "situation_all": ["tool:Bash", "error:timeout*"], "hint_any": ["topic:ssh"], "reason": "r"},
        {"name": "Bad Name", "situation_all": ["tool:Bash"], "reason": "r"},
        {"name": "uses_label", "situation_all": ["ok:no"], "reason": "r"},
        {"name": "uses_hint_id", "situation_all": ["tool:Bash"], "hint_any": ["hint:insight:1"], "reason": "r"},
        {"name": "too_many", "situation_all": ["tool:a", "tool:b", "tool:c", "tool:d", "tool:e"], "reason": "r"},
        {"name": "code_injection", "situation_all": ["tool:Bash; import os"], "reason": "r"},
    ]})
    assert [r["name"] for r in ok] == ["ssh_timeout_with_key_hint"]
    assert len(why) == 5
    assert fw.check("not json") == ([], ["no rules list"])


def test_matching_with_wildcards_and_hint_tags():
    r = {"name": "x", "situation_all": ["tool:Bash", "error:timeout*"], "hint_any": ["topic:ssh"]}
    assert fw.matches(r, {"tool:Bash", "error:timeout_after_30s"}, {"topic:ssh", "host:web1"})
    assert not fw.matches(r, {"tool:Bash"}, {"topic:ssh"})
    assert not fw.matches(r, {"tool:Bash", "error:timeout"}, {"topic:dns"})
    assert fw.matches({**r, "hint_any": []}, {"tool:Bash", "error:timeout"}, set())


def _world(n=2400, seed=3):
    """Synthetic pairs: a hint with topic:ssh helps exactly when the action is Bash with a timeout error — an interaction no single tag
    feature carries. Returns observations in learning's form plus the hint objects."""
    rnd = random.Random(seed)
    ssh = g.new_object("insight", "ssh to the build host needs the key file", ["topic:ssh", "tool:Bash"], {"source": "t"})
    dns = g.new_object("insight", "dns answers slowly on monday", ["topic:dns", "tool:Bash"], {"source": "t"})
    obs = []
    for i in range(n):
        h = rnd.choice([ssh, dns])
        err = rnd.choice(["error:timeout_x", "error:denied"])
        f = {"tool:Bash": 1.0, err: 1.0, f"hint:{h.id}": 1.0}
        p = 0.92 if (err.startswith("error:timeout") and h is ssh) else 0.55
        obs.append((f, int(rnd.random() < p), f"e{i}", 1.0))
    return obs, ssh, dns


def test_offline_keeps_a_rule_that_carries_signal_and_drops_a_useless_one(gd):
    obs, _ssh, _dns = _world()
    good = {"name": "ssh_hint_on_timeouts", "situation_all": ["tool:Bash", "error:timeout*"], "hint_any": ["topic:ssh"]}
    useless = {"name": "dns_hint_on_denied_monday", "situation_all": ["tool:Bash", "hour:03"], "hint_any": ["topic:dns"]}
    m = fw.offline(obs, [good, useless])
    assert m["better"] and m["paired_logloss"] < -2 * m["se"]
    assert m["kept"] == ["ssh_hint_on_timeouts"]  # the useless rule never fires, its weight stays ~0
    assert fw.offline(obs, [useless])["better"] is False


def test_run_asks_the_model_with_training_numbers_only_and_stores_every_rule(gd, monkeypatch):
    obs, ssh, _dns = _world()
    monkeypatch.setattr(learning, "weighted_observations", lambda *a, **k: obs)
    seen = []

    def ask(system, user):
        seen.append(user)
        return json.dumps({"rules": [
            {"name": "ssh_hint_on_timeouts", "situation_all": ["tool:Bash", "error:timeout*"], "hint_any": ["topic:ssh"], "reason": "keys"},
            {"name": "never_fires", "situation_all": ["tool:Nothing"], "reason": "noise"}]})
    monkeypatch.setattr(reflection, "ask_fn", ask)
    r = fw.run("test")
    assert r["state"] == "measured" and r["kept"] == ["ssh_hint_on_timeouts"]
    card = json.loads(seen[0].split("Card:\n", 1)[1])
    assert card["pairs_in_training"] == len(obs) - int(len(obs) * 0.2)  # numbers from the training part only
    with g.lock:
        rows = dict(g.db().execute("SELECT name, state FROM feature_rule").fetchall())
    assert rows == {"ssh_hint_on_timeouts": "active", "never_fires": "offline_rejected"}
    # the kept rule is now a feature of the router for every matching pair
    f = learning.hint_features(ssh.id, ["tool:Bash", "error:timeout_after_30s"])
    assert f.get("rule:ssh_hint_on_timeouts") == 1.0
    assert "rule:ssh_hint_on_timeouts" not in learning.hint_features(ssh.id, ["tool:Bash", "error:denied"])
    assert "active 1" in fw.line() and "offline_rejected 1" in fw.line()
    assert fw.run("test")["state"] in ("no_new_rules", "measured")  # a name is never proposed twice


def test_without_rules_nothing_changes(gd):
    h = g.new_object("insight", "x", ["topic:ssh"], {"source": "t"})
    assert not any(k.startswith("rule:") for k in learning.hint_features(h.id, ["tool:Bash"]))
    assert "no rules yet" in fw.line()


def test_no_model_means_skipped(gd, monkeypatch):
    from vi_mcp import classify
    monkeypatch.setattr(reflection, "ask_fn", None)
    monkeypatch.setattr(classify, "_config", lambda: {})
    assert fw.run("test")["state"] == "skipped"
