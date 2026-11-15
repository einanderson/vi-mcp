"""Three network bugs reported from a Windows installation (0.66.0): the network token in the agent's conversation, a peer address
"new" that is no host, and the standard network reported as "not recorded" on a server that never set it up."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from vi_mcp import config, logbook as lb, net_cli, net_service as nd, p2p, tools  # noqa: E402
from vi_mcp import memory as g  # noqa: E402

from test_netz import net  # noqa: E402,F401


# ---- 3. the network token never enters an agent's conversation ----

def test_net_token_tool_stores_the_token_and_shows_only_its_id(gd):
    out = tools.call_tool("vi_net_token", {}, "a", False)
    token = p2p._token("a")
    assert token and token not in out  # the bug: the token itself was the tool result
    assert p2p.token_id(token) in out and "netz_token" in out and p2p.token_location() in out  # id and where it is kept
    from vi_mcp import db as dbmod
    assert (str(gd) in out) != dbmod.is_url(g.fpath_active)  # the SQLite file is named; a PostgreSQL URL never (it can carry a password)
    again = tools.call_tool("vi_net_token", {}, "a", False)  # a second call names the same token instead of refusing
    assert token not in again and p2p.token_id(token) in again and p2p._token("a") == token


def test_net_forget_takes_the_stored_token(gd):
    o = g.new_object("erkenntnis", "Borg needs BORG_PASSCOMMAND for unattended runs", ["thema:backup"], {"quelle": "test"}, instance="a")
    p2p.generate_token("a")
    out = tools.call_tool("vi_net_forget", {"ids": [o.id], "reason": "outdated"}, "a", False)  # no token from the agent
    assert out.startswith("1 entries recalled")
    with pytest.raises(tools.Rejected, match="does not match"):  # the old way still checks a token that is given
        tools.call_tool("vi_net_forget", {"ids": [o.id], "reason": "x", "token": "wrong"}, "a", False)
    assert tools.call_tool("vi_net_forget", {"ids": [o.id], "reason": "again", "token": p2p._token("a")}, "a", False).startswith("0 entries")
    with pytest.raises(tools.Rejected, match="no network token"):
        tools.call_tool("vi_net_forget", {"all": True, "reason": "x"}, "b", False)


def _main(monkeypatch):
    spec = importlib.util.spec_from_file_location("vi_mcp.__main__", Path(tools.__file__).with_name("__main__.py"))
    main = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "argv", ["vi-mcp"])
    with pytest.raises(SystemExit):
        spec.loader.exec_module(main)
    return main


def test_net_token_cli_shows_id_exports_0600_and_recalls_without_printing_the_token(tmp_path, monkeypatch, capsys):
    db = tmp_path / "vi.db"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"port": 8799, "db": str(db), "instances": {}}))
    monkeypatch.setenv("VI_MCP_CONFIG", str(cfg))
    g.open_db(db)
    o = g.new_object("erkenntnis", "Borg needs BORG_PASSCOMMAND for unattended runs", ["thema:backup"], {"quelle": "test"}, instance="a")
    token = p2p.generate_token("a")
    g.close_db()
    main = _main(monkeypatch)
    assert main._cli(["net-token", "a"]) == 0
    out = capsys.readouterr().out
    assert token not in out and p2p.token_id(token) in out
    assert main._cli(["net-token", "a", "--forget", "--all", "--reason", "leaving the network"]) == 0
    out = capsys.readouterr().out
    assert token not in out and "1 entries recalled" in out
    exp = tmp_path / "backup" / "token"
    assert main._cli(["net-token", "a", "--export", str(exp)]) == 0
    assert token not in capsys.readouterr().out and exp.read_text().strip() == token
    if os.name != "nt":
        assert exp.stat().st_mode & 0o777 == 0o600
    assert main._cli(["net-token", "nobody"]) == 1
    g.open_db(db)
    assert g.db().execute("SELECT COUNT(*) FROM net_revocation WHERE oid = ?", (o.id,)).fetchone()[0] == 1
    g.close_db()


# ---- 4. a peer address must name a real host ----

@pytest.mark.parametrize("address", ["https://new/vi-p2p/wiki", "https://new", "new", "ftp://wiki.example.org", "https://wiki.example.org?x=1",
                                     "https://user@wiki.example.org", "https://wiki.example.org:99999", "https://-bad-.example.org",
                                     "https://wiki.123", "https:///vi-p2p/wiki", "https://wiki example.org"])
def test_invalid_peer_addresses_are_named(address):
    assert lb.address_problem(address)


@pytest.mark.parametrize("address", ["https://vi-wiki.org/vi-p2p/wiki", "https://wiki.example.org:8443/a/b", "http://127.0.0.1:8766",
                                     "http://localhost:1", "https://[::1]:8766", "https://10.0.0.5/vi-p2p/firma", "https://xn--bcher-kva.example"])
def test_valid_peer_addresses_pass(address):
    assert lb.address_problem(address) is None


def test_wiki_start_refuses_a_template_word():
    with pytest.raises(ValueError, match="not a host name"):
        net_cli.wiki_start("new")  # the bug: became https://new/vi-p2p/wiki


def test_the_msi_dialog_no_longer_offers_new_as_a_word_to_type():
    ui = (Path(tools.__file__).parents[1] / "deploy" / "windows" / "ui.sql").read_text(encoding="utf-8")
    label = next(line for line in ui.splitlines() if "'WikiLabel'" in line)
    assert "new:" not in label  # "empty = as before, new: vi-wiki.org" read as: type new for vi-wiki.org
    ps = (Path(tools.__file__).parents[1] / "deploy" / "windows" / "install.ps1").read_text(encoding="utf-8")
    assert '"new"' in ps  # a remembered or typed "new" from an older dialog is read as the default server


def test_logbook_refuses_invalid_peers_and_service_reports_existing_ones(net, tmp_path):
    seed, pub = net.key_()
    a = net.build(seed, [pub], start=["https://new/vi-p2p/wiki", "https://ok.example/vi-p2p/wiki"])
    assert {p["adresse"] for p in a.log.peers()} == {"https://ok.example/vi-p2p/wiki"}  # the start word is not remembered
    with a.log.lock:  # an entry from an older version, before addresses were checked
        a.log.db.execute("INSERT INTO peer (address, source) VALUES (?,?)", ("https://old/vi-p2p/wiki", "start"))
    assert "https://old/vi-p2p/wiki" not in a.log.choose_peers(10)  # never contacted
    snap = {p["adresse"]: p for p in a.snapshot()["peers"]}
    assert snap["https://old/vi-p2p/wiki"]["state"] == "invalid" and "not a host name" in snap["https://old/vi-p2p/wiki"]["reason"]
    rejected = a.snapshot()["rejected_addresses"]
    assert rejected and rejected[0]["address"] == "https://new/vi-p2p/wiki" and rejected[0]["source"] == "start"


def test_peer_is_paused_after_repeated_failures_with_reason_and_resume_time(net):
    seed, pub = net.key_()
    a = net.build(seed, [pub], start=["https://down.example/vi-p2p/wiki"])
    addr = "https://down.example/vi-p2p/wiki"
    for _ in range(lb.PEER_PAUSE_AFTER - 1):
        a.log.peer_result(addr, False, "URLError: name not resolved")
    p = next(x for x in a.snapshot()["peers"] if x["adresse"] == addr)
    assert p["state"] == "failing"
    a.log.peer_result(addr, False, "URLError: name not resolved")
    p = next(x for x in a.snapshot()["peers"] if x["adresse"] == addr)
    assert p["state"] == "paused" and "URLError" in p["reason"] and f"{lb.PEER_PAUSE_AFTER} failed attempts" in p["reason"]
    assert p["resume_at"] and p["resume_at"] > time.strftime("%Y-%m-%dT%H:%M", time.gmtime())
    assert addr not in a.log.choose_peers(10)
    a.log.peer_result(addr, True)
    p = next(x for x in a.snapshot()["peers"] if x["adresse"] == addr)
    assert p["state"] == "reachable" and p["resume_at"] is None


def test_peer_removal_by_tool_reaches_the_service_and_stays_removed(gd, net, monkeypatch):
    monkeypatch.setattr(p2p, "config", {"networks": {"wiki": {"kind": "free", "instance": "netz-wiki"}}})
    with pytest.raises(tools.Rejected, match="reason"):
        tools.call_tool("vi_net_peer_remove", {"network": "wiki", "address": "https://old/vi-p2p/wiki"}, "a", False)
    out = tools.call_tool("vi_net_peer_remove", {"network": "wiki", "address": "https://old/vi-p2p/wiki/", "reason": "template word"}, "a", False)
    assert "https://old/vi-p2p/wiki" in out
    with pytest.raises(tools.Rejected, match="own instance"):
        tools.call_tool("vi_net_peer_remove", {"network": "wiki", "address": "https://x.example", "reason": "r"}, config.OPEN, False)
    reply = p2p.outbox("wiki")
    assert reply["removed_peers"] == ["https://old/vi-p2p/wiki"]
    assert p2p.outbox("standard")["removed_peers"] == []  # per network
    seed, pub = net.key_()
    a = net.build(seed, [pub])
    with a.log.lock:
        a.log.db.execute("INSERT INTO peer (address, source) VALUES (?,?)", ("https://old/vi-p2p/wiki", "start"))
    a.mcp.outbox = lambda state=None: {"objekte": [], "widerrufe": [], "removed_peers": reply["removed_peers"]}
    a.outbox()
    assert "https://old/vi-p2p/wiki" not in {p["adresse"] for p in a.log.peers()}
    a.log.peer_remember("https://old/vi-p2p/wiki", "austausch")  # removed stays removed while the memory lists it
    assert "https://old/vi-p2p/wiki" not in {p["adresse"] for p in a.log.peers()}


def test_peers_remove_cli_drops_it_from_the_table_and_the_start_list(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VI_P2P_DIR", str(tmp_path / "p2p"))
    (tmp_path / "p2p").mkdir()
    seed = tmp_path / "p2p" / "schluessel"
    pub = lb.generate_keys(seed)
    conf = {"mode": "peer", "net": "wiki", "kind": "free", "db": str(tmp_path / "p2p" / "netz.db"), "seed": str(seed), "founders": [pub],
            "startpeers": ["https://new/vi-p2p/wiki", "https://vi-wiki.org/vi-p2p/wiki"]}
    (tmp_path / "p2p" / "wiki.json").write_text(json.dumps(conf))
    log = lb.Logbook(conf["db"], seed, [pub])
    with log.lock:
        log.db.execute("INSERT INTO peer (address, source) VALUES (?,?)", ("https://new/vi-p2p/wiki", "start"))
    log.close() if hasattr(log, "close") else None
    monkeypatch.setattr(net_cli, "_service", lambda *a, **k: None)
    assert net_cli.main(["--network", "wiki", "peers", "remove", "https://new/vi-p2p/wiki"]) == 0
    raw = json.loads((tmp_path / "p2p" / "wiki.json").read_text())
    assert raw["startpeers"] == ["https://vi-wiki.org/vi-p2p/wiki"]
    log = lb.Logbook(conf["db"], seed, [pub])
    assert "https://new/vi-p2p/wiki" not in {p["adresse"] for p in log.peers()}
    assert "removed" in capsys.readouterr().out


# ---- 5. a network nobody set up is not "not recorded" ----

def test_status_names_why_a_network_has_no_snapshot(gd, monkeypatch):
    monkeypatch.setattr(p2p, "config", {"networks": {"wiki": {"kind": "free", "instance": "netz-wiki"}}})
    monkeypatch.setattr(p2p, "service_instances", frozenset({"a", "netz-wiki"}))  # no instance p2p: standard was never set up
    st = p2p.status()["netze"]
    assert "not set up" in st["standard"]["stand"] and "p2p" in st["standard"]["stand"]
    assert "never reported" in st["wiki"]["stand"]  # set up, but its service has not called
    tools.call_tool("vi_net_outbox", {"state": {"netz": "wiki", "version": "0.66.0", "peers": []}}, "netz-wiki", False)
    st = p2p.status()["netze"]
    assert st["wiki"]["stand"]["version"] == "0.66.0" and "stale" not in st["wiki"]["stand"]
    with g.lock:
        g.db().execute("UPDATE net_state SET ts = ? WHERE net = 'wiki'", ("2026-11-01T10:00:00+00:00",))
        g.db().commit()
    st = p2p.status()["netze"]["wiki"]["stand"]
    assert "2026-11-01T10:00" in st["stale"] and "vi-p2p" in st["stale"]


def test_service_names_follow_the_platform():
    assert p2p.service_name("standard", windows=False) == "vi-p2p" and p2p.service_name("wiki", windows=False) == "vi-p2p@wiki"
    assert p2p.service_name("standard", windows=True) == "vi-p2p" and p2p.service_name("wiki", windows=True) == "vi-p2p-wiki"
