"""0.68.0: vi_feature_propose — external AIs read the router's card and propose feature rules; the same two-step gate decides in sleep."""
import json

import pytest

from vi_mcp import config, feature_writer as fw, learning, tools
from vi_mcp import memory as g
from vi_mcp.tools import Rejected

from test_feature_writer import _world

GOOD = {"name": "ssh_hint_on_timeouts", "situation_all": ["tool:Bash", "error:timeout*"], "hint_any": ["topic:ssh"], "reason": "keys"}
USELESS = {"name": "never_fires_rule", "situation_all": ["tool:Nothing"], "reason": "noise"}


def test_an_open_instance_without_a_token_cannot_propose(gd):
    with pytest.raises(Rejected):
        tools.call_tool("vi_feature_propose", {"action": "propose", "rules": [GOOD]}, config.OPEN, False)


def test_propose_checks_rules_ignores_numbers_and_limits_open_proposals(gd, monkeypatch):
    bad = {"name": "uses_label", "situation_all": ["ok:yes"], "reason": "x"}
    r = fw.propose("claude-code", "claude", [{**GOOD, "weight": 9.9}, bad, USELESS], "test")
    assert r["stored"] == ["ssh_hint_on_timeouts", "never_fires_rule"] and any("uses_label" in w for w in r["rejected"])
    with g.lock:
        d, state, by = g.db().execute("SELECT definition, state, proposed_by FROM feature_rule WHERE name = 'ssh_hint_on_timeouts'").fetchone()
    assert "9.9" not in d and "weight" not in d and state == "proposed" and by == "claude-code (claude)"
    assert "exists already" in fw.propose("codex", "gpt", [GOOD])["rejected"][0]  # names are unique across proposers
    monkeypatch.setattr(fw, "MAX_PENDING", 3)
    r = fw.propose("claude-code", "claude", [{**USELESS, "name": f"extra_{i}_rule"} for i in range(3)])
    assert len(r["stored"]) == 1 and any("open proposals" in w for w in r["rejected"])
    assert {p["name"] for p in fw.proposals("codex")} == set()  # list shows only the caller's own


def test_sleep_measures_each_proposer_apart_and_activates_only_what_the_gate_keeps(gd, monkeypatch):
    obs, ssh, _dns = _world()
    monkeypatch.setattr(learning, "weighted_observations", lambda *a, **k: obs)
    fw.propose("claude-code", "claude", [GOOD])
    fw.propose("codex", "gpt", [USELESS])
    r = fw.evaluate_proposed()
    assert r["state"] == "measured" and set(r["by_proposer"]) == {"claude-code (claude)", "codex (gpt)"}
    states = {p["name"]: p["state"] for p in fw.proposals()}
    assert states == {"ssh_hint_on_timeouts": "active", "never_fires_rule": "offline_rejected"}
    assert learning.hint_features(ssh.id, ["tool:Bash", "error:timeout_after_30s"]).get("rule:ssh_hint_on_timeouts") == 1.0
    assert fw.evaluate_proposed()["state"] == "nothing_proposed"


def test_the_card_through_the_tool_has_weights_and_the_rule_language_but_no_holdout(gd, monkeypatch):
    obs, _ssh, _dns = _world()
    monkeypatch.setattr(learning, "weighted_observations", lambda *a, **k: obs)
    fw._card_cache.update(at=0.0, db=None, card=None)
    c = json.loads(tools.call_tool("vi_feature_propose", {"action": "card"}, "claude-code", False, family="claude"))
    assert c["strongest_weights"] and c["rule_language"]["numbers"].startswith("none")
    assert c["pairs_in_training"] == len(obs) - int(len(obs) * 0.2)
    out = json.loads(tools.call_tool("vi_feature_propose", {"action": "propose", "rules": [GOOD]}, "claude-code", False, family="claude"))
    assert out["stored"] == ["ssh_hint_on_timeouts"]
    assert json.loads(tools.call_tool("vi_feature_propose", {"action": "list"}, "claude-code", False))[0]["state"] == "proposed"
