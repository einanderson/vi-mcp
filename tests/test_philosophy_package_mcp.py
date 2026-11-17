"""0.69.6: a philosophy is handed over MCP (install with package=…), read back whole (export) and used by name once installed — before,
install took only a built-in name or a directory on the server, and an installed philosophy could not be read or bound by its name."""
import json

import pytest

pytest.importorskip("clingo")

from vi_mcp import tools  # noqa: E402
from vi_mcp import memory as g  # noqa: E402


def _copy(name="messung-kopie"):
    parts = json.loads(tools.call_tool("vi_philosophy", {"action": "export", "name": "messung"}, "a", False))
    assert set(parts) == {"manifest", "seed", "worlds"} and parts["worlds"]
    parts["manifest"]["name"] = name
    return parts


def test_install_from_a_package_then_read_export_and_bind_by_name(gd):
    out = tools.call_tool("vi_philosophy", {"action": "install", "package": _copy()}, "a", False)
    assert out.startswith("installed ") and "messung-kopie" in out
    assert "messung-kopie" in tools.call_tool("vi_philosophy", {"action": "list"}, "a", False)
    assert "messung-kopie" in tools.call_tool("vi_philosophy", {"action": "show", "name": "messung-kopie"}, "a", False)
    back = json.loads(tools.call_tool("vi_philosophy", {"action": "export", "name": "messung-kopie"}, "a", False))
    assert back["seed"] == _copy()["seed"] and back["worlds"] == _copy()["worlds"]
    assert "bound" in tools.call_tool("vi_philosophy", {"action": "use", "name": "messung-kopie"}, "a", False).lower()


def test_worlds_as_jsonl_text_and_the_checks_still_decide(gd):
    p = _copy("text-worlds")
    p["worlds"] = "\n".join(json.dumps(w) for w in p["worlds"])
    assert tools.call_tool("vi_philosophy", {"action": "install", "package": p}, "a", False).startswith("installed ")
    bad = _copy("broken")
    bad["seed"] = "# P1\nresult(X) <= ~known(X)"
    with pytest.raises(tools.Rejected, match="package refused"):
        tools.call_tool("vi_philosophy", {"action": "install", "package": bad}, "a", False)


def test_names_are_owned(gd):
    with pytest.raises(tools.Rejected, match="built-in philosophy"):
        tools.call_tool("vi_philosophy", {"action": "install", "package": _copy("messung")}, "a", False)
    tools.call_tool("vi_philosophy", {"action": "install", "package": _copy("mine")}, "a", False)
    with pytest.raises(tools.Rejected, match="installed by a"):
        tools.call_tool("vi_philosophy", {"action": "install", "package": _copy("mine")}, "b", False)
    with pytest.raises(tools.Rejected, match="neither built in"):
        tools.call_tool("vi_philosophy", {"action": "show", "name": "unknown-one"}, "a", False)


def test_which_philosophy_fits_is_declared_measured_and_learned(gd):
    from vi_mcp import intuition, philosophy_fit
    p = _copy("incident-calm")
    p["manifest"]["situations"] = ["topic:incident", "task:triage"]
    tools.call_tool("vi_philosophy", {"action": "install", "package": p}, "a", False)
    out = tools.call_tool("vi_philosophy", {"action": "suggest", "situation": "an incident on the mail relay"}, "a", False)
    assert "topic:incident" in out and "- incident-calm" in out and "declared" in out
    # measured: messung helps in incidents, incident-calm does not — after enough ratings messung leads
    for _ in range(8):
        tools.call_tool("vi_philosophy", {"action": "rate", "name": "messung", "tags": ["topic:incident"], "helped": True,
                                          "reason": "only measured evidence closed the incident"}, "a", False)
        tools.call_tool("vi_philosophy", {"action": "rate", "name": "incident-calm", "tags": ["topic:incident"], "helped": False,
                                          "reason": "its verdict ignored the measurements"}, "a", False)
    import random
    firsts = [philosophy_fit.suggest("incident", None, "a", random.Random(i))["ranking"][0]["name"] for i in range(20)]
    assert firsts.count("messung") >= 18
    assert {"situation": "topic:incident", "philosophy": "messung", "helped": 8.0, "not_helped": 0.0} in philosophy_fit.pairs()
    with pytest.raises(tools.Rejected, match="reason needed"):
        tools.call_tool("vi_philosophy", {"action": "rate", "name": "messung", "tags": ["topic:incident"], "helped": True, "reason": "ok"}, "a", False)
    # in the prompt context: once per conversation
    h = intuition.prompt_hints("the incident on the mail relay again", "a", [])
    line = next(x for x in h["hinweise"] if x.startswith("[Philosophy]"))
    assert "messung" in line and "vi_philosophy(action=\"evaluate\"" in line and "philosophy:messung" in h["ids"]
    again = intuition.prompt_hints("the incident on the mail relay again", "a", h["ids"])
    assert not any(x.startswith("[Philosophy] This situation fits the philosophy messung") for x in again["hinweise"])


def test_situations_are_checked_and_a_situation_nobody_declared_gets_a_reason(gd):
    bad = _copy("bad-situations")
    bad["manifest"]["situations"] = ["incident"]
    with pytest.raises(tools.Rejected, match="situations: a list of kind:value"):
        tools.call_tool("vi_philosophy", {"action": "install", "package": bad}, "a", False)
    out = tools.call_tool("vi_philosophy", {"action": "suggest", "situation": "bake a cake"}, "a", False)
    assert "no situation tag" in out
