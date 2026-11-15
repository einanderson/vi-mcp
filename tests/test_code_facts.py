"""Structure facts of code and the philosophy package `code` (Unreleased): the importer's ast pass derives calls, imports, table
writes, raises, tests, history and literal constants; they are stored as fact objects and read by clingo-object rules through the one
object path (logic.object_data → vi__tag). The bridge over stored objects must give exactly the facts of a direct scan."""
import subprocess
import textwrap
from pathlib import Path

import pytest

from vi_mcp import code_facts as cf
from vi_mcp import code_import as ci
from vi_mcp import logic, tools

REPO = Path(__file__).resolve().parent.parent

STORE = textwrap.dedent('''\
    """A store."""
    from . import db as g
    from .db import commit
    import json

    ALIASES = {"old_name": "new_name", "older": "new_name"}
    REGISTRY = [{"name": "new_name", "doc": "x y"}, {"name": "other"}]


    class Broken(Exception):
        pass


    def save(row):
        """Writes under the lock."""
        with g.lock:
            g.conn().execute("INSERT INTO object (id) VALUES (?) ON CONFLICT(id) DO UPDATE SET id = excluded.id", (row,))
            commit()
        return json.dumps(row)


    def wipe(row):
        g.conn().execute("DELETE FROM object WHERE id = ?", (row,))
        if not row:
            raise Broken("empty")
        raise ValueError("never")


    class Store:
        def put(self, row):
            from . import late
            late.touch()
            return self.get(row) or save(row)

        def get(self, row):
            return helper(row)


    def helper(row):
        x = [row, row, row]
        y = sorted(x)
        return {"a": x, "b": y, "c": len(x)}
    ''')

DB = 'import threading\nfrom . import store\nlock = threading.RLock()\n\n\ndef conn():\n    return None\n\n\ndef commit():\n    pass\n'
COPY = STORE.split("def helper(row):")[1]
OTHER = "def helper2(row):" + COPY + "\n\ndef only_here():\n    return 1\n"
TEST = textwrap.dedent('''\
    from pkg import store


    def test_save():
        assert store.save(1)
    ''')


def _git(repo, *a):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "shop"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "pkg" / "__init__.py").write_text('"""The shop."""\n')  # an empty file has no piece, so a server without facts never knows it
    (r / "pkg" / "store.py").write_text(STORE)
    (r / "pkg" / "db.py").write_text(DB)
    (r / "pkg" / "late.py").write_text("def touch():\n    return 1\n")
    (r / "pkg" / "other.py").write_text(OTHER)
    (r / "tests" / "test_store.py").write_text(TEST)
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "one")
    return r


def _fn(st, name):
    return next(f for f in st["funktionen"] if f["name"] == name)


def test_structure_of_one_file():
    mods = {"pkg", "pkg.store", "pkg.db", "pkg.late"}
    st = cf.structure("pkg/store.py", STORE, mods)
    assert st["modul"] == "pkg.store" and st["importiert"] == ["json", "pkg.db"]
    assert st["importiert_spaet"] == ["pkg.late"]  # inside a function: no import-time edge
    save = _fn(st, "pkg.store.save")
    assert save["schreibt"] == ["object"] and save["schreibt_ungesperrt"] == []  # DO UPDATE SET names no table
    assert "pkg.db.conn" in save["ruft"] and "pkg.db.commit" in save["ruft_gesperrt"]
    wipe = _fn(st, "pkg.store.wipe")
    assert wipe["schreibt_ungesperrt"] == ["object"] and wipe["wirft"] == ["ValueError", "pkg.store.Broken"]
    put = _fn(st, "pkg.store.Store.put")
    assert {"pkg.late.touch", "pkg.store.Store.get", "pkg.store.save"} <= set(put["ruft"])  # late import, self, module function
    assert _fn(st, "pkg.store.helper")["koerper"] and _fn(st, "pkg.store.Store.get")["koerper"] is None  # a one-liner is no duplicate
    consts = {c["name"]: c for c in st["konstanten"]}
    assert consts["pkg.store.ALIASES"]["schluessel"] == ["old_name", "older"] and consts["pkg.store.ALIASES"]["wert"] == ["new_name"]
    assert consts["pkg.store.REGISTRY"]["wert"] == ["new_name", "other"]  # a list of records: the name values
    assert cf.structure("x.py", "def (:", mods) is None
    assert cf.is_test("tests/test_store.py", "tests.test_store.test_save") and not cf.is_test("pkg/store.py", "pkg.store.test_x")


def test_history_from_blame(repo):
    (repo / "pkg" / "late.py").write_text("def touch():\n    return 2\n")
    import os
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "two"], check=True,
                   env={**os.environ, "GIT_COMMITTER_DATE": "2030-01-01T00:00:00+00:00"})  # newer than "one" even within one second
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    st = cf.scan(repo, history=True)
    assert _fn(st["pkg/late.py"], "pkg.late.touch")["geaendert_in"][0] == head[:12]  # newest first
    assert len(_fn(st["pkg/late.py"], "pkg.late.touch")["geaendert_in"]) == 2
    (repo / "pkg" / "late.py").write_text("def touch():\n    return 3\n")  # uncommitted: blame has no commit for it
    assert cf.blame(repo, "pkg/late.py").get(2) is None


def _call(tool, args):
    return tools.call_tool(tool, args, "a", False)


def _bridged(repo_name: str) -> set[str]:
    show = "\n".join(f"seen_{p}(" + ", ".join(f"X{i}" for i in range(n)) + f") <= {p}(" + ", ".join(f"X{i}" for i in range(n)) + ")"
                     for p, n in PREDS.items())
    r = logic._run(cf.bridge(repo_name) + show)
    atoms = set().union(*[set(m) for m in r["models"]])
    return {a[5:] for a in atoms if a.startswith("seen_")}


PREDS = {"snapshot": 1, "module": 1, "imports": 2, "imports_late": 2, "defines": 2, "in_module": 2, "calls": 2, "calls_locked": 2,
         "writes_table": 2, "writes_unlocked": 2, "raises": 2, "test": 1, "body": 2, "changed_in": 2, "const_key": 2, "const_value": 2}


def test_stored_facts_read_through_the_object_path_equal_the_direct_scan(gd, repo):
    pytest.importorskip("clingo")
    r = ci.run(repo, "shop", _call)
    assert r["fakten_objekte"] > 0 and r["ohne_fakten"] == 0
    direct = {"".join(a.split()) for a in cf.direct_facts(cf.scan(repo, history=True), "shop")}
    assert _bridged("shop") == direct  # one meaning, two routes
    st = ci.server_state("shop", "shop")
    assert set(st["mit_fakten"]) == {p for p in st["dateien"] if p.endswith(".py")}


def test_facts_are_backfilled_once_and_follow_changes(gd, repo, monkeypatch):
    real = cf.structure
    monkeypatch.setattr(cf, "structure", lambda *a: None)  # as if the importer had no facts yet: text pieces only
    ci.run(repo, "shop", _call)
    monkeypatch.setattr(cf, "structure", real)
    assert ci.server_state("shop", "shop")["mit_fakten"] == []
    again = ci.run(repo, "shop", _call)  # nothing changed, but no Python file has facts: each is sent once
    assert again["geaendert"] == 6 and again["fakten_objekte"] > 0
    assert ci.run(repo, "shop", _call)["geaendert"] == 0
    (repo / "pkg" / "other.py").write_text("def only_here():\n    return 1\n")
    _git(repo, "commit", "-qam", "drop helper2")
    ci.run(repo, "shop", _call)
    from vi_mcp import memory as g
    gone = g.load(ci.piece_id("shop", "shop", "pkg/other.py", "#funktion:helper2"))
    assert gone.mode == "archived"  # nothing disappears silently


def test_an_old_server_gets_no_facts_and_no_resend(gd, repo, monkeypatch):
    sent = []

    def old(tool, args):
        if args.get("action") == "state":
            st = ci.server_state(args["index"], args["repo"])
            st.pop("mit_fakten")  # a server before the structure facts
            return __import__("json").dumps(st)
        sent.append(args)
        return _call(tool, args)

    ci.run(repo, "shop", old)
    assert sent and not any("struktur" in f for a in sent for f in a["files"])
    sent.clear()
    assert ci.run(repo, "shop", old)["geaendert"] == 0 and not sent


def test_fact_objects_are_code_never_hints_vectors_or_training(gd, repo):
    from vi_mcp import vector
    from vi_mcp import memory as g
    ci.run(repo, "shop", _call)
    facts = [o for o in ci._held("shop", "shop") if ci._is_fact(o)]
    assert facts and all("art_wissen:code" in o.hashtags and o.visibility == "internal" for o in facts)
    assert vector.FACT_TAGS == cf.FACT_TAGS
    line, _p = ci.file_line("shop", str(repo / "pkg" / "store.py"))
    assert "None" not in line and "save" in line  # the file line names pieces, not fact objects
    with g.lock:
        rows = g.db().execute("SELECT o.id FROM object o WHERE o.id NOT IN (SELECT object_id FROM tag WHERE tag IN (?,?,?))",
                              cf.FACT_TAGS).fetchall()
    assert len(rows) < len(ci._held("shop", "shop"))


# ── the philosophy package `code` ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_package_passes_its_checks():
    pytest.importorskip("clingo")
    from clingo_object import philosophy as ph
    r = ph.check(ph.load("code"))
    assert r["ok"], r["errors"]
    assert r["coverage"]["status"] == {"ok": True, "violated": True}


@pytest.mark.parametrize("mutation", [(" & ~is_excused(Q, X)", ""), (" & ~covered(F)", ""), (" & (M != N)", ""),
                                      (" & ~registered(G, E)", ""),
                                      ("clean(Q) <= rule(Q) & snapshot(R) & ~has_finding(Q)", "clean(Q) <= rule(Q) & ~has_finding(Q)")])
def test_every_mutation_of_the_seed_is_refused_by_a_world(mutation):
    pytest.importorskip("clingo")
    from clingo_object import philosophy as ph
    pkg = ph.load("code")
    assert mutation[0] in pkg["seed"]
    pkg["seed"] = pkg["seed"].replace(mutation[0], mutation[1])
    assert not ph.check(pkg)["ok"]


def test_the_repository_holds_its_own_architecture_rules():
    """This repository against its profile: no import cycle, guarded writes under the lock, every tool alias resolves. Without the
    exemption of _migrate the lock rule must find it — the profile's terms are the importer's (strings), not constants."""
    pytest.importorskip("clingo")
    from clingo_object import philosophy_code as pc
    st = cf.scan(REPO)
    rep = cf.report(cf.evaluate(st, "vi-mcp", pc.PROFILE_VI_MCP))
    assert rep["state"] == "erfuellt" and not rep["violations"]
    assert {q: rep["status"][q] for q in ("no_import_cycle", "locked_writes", "english_tools")} == dict.fromkeys(
        ("no_import_cycle", "locked_writes", "english_tools"), "ok")
    assert ("locked_writes", "vi_mcp.memory._migrate") in {(q, x) for q, x, _w in rep["excused"]}
    loose = "\n".join(x for x in pc.PROFILE_VI_MCP.splitlines() if "_migrate" not in x)
    assert ("locked_writes", "vi_mcp.memory._migrate", "object") in cf.report(cf.evaluate(st, "vi-mcp", loose))["findings"]
    dangling = cf.report(cf.evaluate(st, "vi-mcp", pc.PROFILE_VI_MCP, 'const_value("vi_mcp.api_en.TOOL_ALIASES", "vi_nowhere")'))
    assert ("english_tools", "vi_nowhere", "vi_mcp.api_en.TOOL_ALIASES") in dangling["findings"]


def test_the_package_runs_in_the_vi_over_stored_facts(gd, repo):
    pytest.importorskip("clingo")
    from clingo_object import philosophy_code as pc
    ci.run(repo, "shop", _call)
    prof = 'guarded("object")\nalias_of("pkg.store.ALIASES", "pkg.store.REGISTRY")\n'
    r = logic._run(cf.bridge("shop") + prof + pc.SEED)  # the raw models: check() cuts its list of conclusions for display
    atoms = set.intersection(*[set(m) for m in r["models"]])
    assert 'finding(locked_writes,"pkg.store.wipe")' in atoms and 'status(locked_writes,violated)' in atoms
    assert 'finding(no_duplicates,"pkg.store.helper")' in atoms and 'grund(no_duplicates,"pkg.other.helper2","pkg.store.helper")' in atoms
    assert 'tested_by("pkg.store.save","tests.test_store.test_save")' in atoms and 'untested("pkg.store.wipe")' in atoms
    assert 'status(english_tools,ok)' in atoms and 'status(no_import_cycle,violated)' in atoms  # pkg.db imports pkg.store at load


def test_training_pairs_come_from_the_facts_and_pass_the_one_definition():
    from vi_mcp import qa_pairs as qa
    st = cf.structure("pkg/store.py", STORE, {"pkg", "pkg.store", "pkg.db", "pkg.late"})
    wipe = _fn(st, "pkg.store.wipe")
    r = cf.training_pairs(wipe, "pkg/store.py", "erkenntnis:code-x")
    assert r["schema"] == qa.SCHEMA_VERSION and r["generator"] == "code-facts-v1" and r["rejected"] == {}
    every = r["pairs"] + [r["probe"]]
    doc = cf.training_document(wipe, "pkg/store.py")
    assert all(qa.check(p, doc) is None for p in every)  # each pair is grounded in the function's own facts
    text = " ".join(p["answer"] for p in every)
    assert "object outside a lock" in text and "pkg.store.Broken" in text and "pkg.db.conn" in text
    assert r["probe"] not in r["pairs"]  # the probe is never trained
    assert cf.training_pairs(wipe, "pkg/store.py", "erkenntnis:code-x") == r  # reproducible: the probe depends on the id only
