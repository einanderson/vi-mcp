"""Virtual machine contract: dynamic state, isolation, restoration and checked program writing."""
import json

import pytest

pytest.importorskip("clingo")
from vi_mcp import logic, machine, tools
from vi_mcp import memory as g

PROGRAM = 'ans(P) <= (O in objects("service:api")) & (O.port == P)'


def call(action="status", owner="a", name="test", **kw):
    return json.loads(tools.call_tool("vi_machine", {"action": action, "name": name, **kw}, owner, False))


def fact(port=80, **kw):
    return call("execute", instruction="tell.fact api", payload={"text": "API endpoint", "hashtags": ["service:api", f"port:{port}"],
                                                                 "source": "independent inventory"}, **kw)


def run(**kw):
    return call("execute", instruction="ask.rule", payload={"source": PROGRAM}, **kw)


def test_dynamic_data_changes_results_without_changing_program(gd):
    call("create")
    fact()
    first = run()
    assert first["answers"] == ['ans("80")'] and first["complete"]
    fact(8080)
    second = run()
    assert second["answers"] == ['ans("8080")'] and second["state_hash"] != first["state_hash"]
    assert first["source"] == second["source"] == PROGRAM


def test_snapshot_restore_fork_and_process_restart(gd):
    call("create")
    fact()
    before = call()
    call("snapshot", snapshot="initial")
    call("fork", target="fork", snapshot="initial")
    fact(8080)
    assert run()["answers"] == ['ans("8080")'] and run(name="fork")["answers"] == ['ans("80")']
    call("restore", snapshot="initial")
    assert call()["state_hash"] == before["state_hash"]
    g.close_db()
    g.open_db(gd)
    assert run()["answers"] == ['ans("80")']
    with pytest.raises(tools.Rejected, match="already exists"):
        call("snapshot", snapshot="initial")


def test_ownership_and_revision_are_enforced(gd):
    call("create")
    fact()
    with pytest.raises(tools.Rejected, match="unknown"):
        call(owner="b")
    call("create", owner="b")
    assert run(owner="b")["answers"] == []
    with pytest.raises(tools.Rejected, match="stale"):
        fact(8080, expected_revision=0)
    assert run()["answers"] == ['ans("80")']
    with pytest.raises(tools.Rejected):
        call("create", owner="offen")


def test_def_rule_search_proposal_and_tombstone(gd):
    call("create")
    fact()
    call("execute", instruction="def.rule endpoint", payload={"source": PROGRAM})
    assert call("execute", instruction="ask.rule endpoint")["answers"] == ['ans("80")']
    found = call("execute", instruction="ask.what service:api")
    assert [o["id"] for o in found["objects"]] == ["api"]
    p = call("execute", instruction="do remove_everything", payload={"command": "untrusted"})
    assert p["state"] == "proposed" and run()["answers"] == ['ans("80")']
    call("execute", instruction="tell.fact api", payload={"deleted": True, "reason": "service removed"})
    assert run()["answers"] == []
    assert machine._load("a", "test").data["zustand"]["objects"]["api"]["deleted"]


def test_model_frontend_uses_own_world_and_checks_generated_program(gd, monkeypatch):
    call("create")
    fact(8080)
    # A global object must never enter this machine's world.
    g.new_object("erkenntnis", "foreign API", ["service:api", "port:9999"], {"quelle": "fixture"})
    replies = iter(['import os', '```logic\n' + PROGRAM + '\n```'])
    monkeypatch.setattr(logic, "agent_llm", lambda messages: next(replies))
    result = call("answer", question="Which port does the API use?")
    assert result["answers"] == ['ans("8080")'] and result["rounds"] == 2
    checked = call("execute", instruction="rate " + result["receipt"],
                   payload={"expected_answers": ['ans("8080")'], "reference": "independent fixture", "question": result["question"]})
    assert checked["correct"]


def test_rejected_program_and_credentials_do_not_change_state(gd):
    call("create")
    before = call()
    for instruction, payload in (("def.rule evil", {"source": "import os"}),
                                  ("tell.fact secret", {"text": "password=secret123", "source": "fixture"})):
        with pytest.raises(tools.Rejected):
            call("execute", instruction=instruction, payload=payload)
    assert call() == before


def test_machine_and_snapshot_are_never_network_exports(gd):
    from vi_mcp import dataset
    call("create")
    fact()
    call("snapshot", snapshot="initial")
    assert all(o.visibility == "privat" for o in g.load_many([machine._id("a", "test"), machine._id("a", "initial", True)]).values())
    assert not dataset.build(agentic=False)["qa"]


def test_import_refresh_and_reference_training_publication(gd, tmp_path, monkeypatch):
    from vi_mcp import dataset, evaluation
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path / "eval")
    src = g.new_object("erkenntnis", "API endpoint", ["service:api", "port:80"], {"quelle": "inventory"})
    g.new_object("erkenntnis", "private foreign API", ["service:api", "port:9999"], {"quelle": "fixture"}, visibility="privat", instance="b")
    call("create")
    call("import", filters=["service:api"])
    assert run()["answers"] == ['ans("80")']
    src.hashtags.remove("port:80")
    src.hashtags.append("port:8080")
    g.save(src)
    assert run()["answers"] == ['ans("80")']  # imported world is frozen until refreshed
    call("import", filters=["service:api"])
    ex = run()
    assert ex["answers"] == ['ans("8080")']
    assert not any(r["meta"].get("kind") == "machine" for r in dataset.build()["logic"])
    call("execute", instruction="rate " + ex["receipt"], payload={"expected_answers": ['ans("8080")'],
         "reference": "independent fixture", "question": "Which port does the API use?"})
    call("teach", receipt=ex["receipt"])
    data = dataset.build()
    positive = [r for r in data["logic"] if r["meta"].get("kind") == "machine"]
    assert len(positive) == 1 and positive[0]["meta"]["validation"] == "reference"
    assert all(row["meta"]["id"] != positive[0]["meta"]["id"] for row in data["qa"])
    assert "9999" not in json.dumps(positive)
    # A real model mistake is a KTO negative, using the same independent reference answer.
    wrong = call("execute", instruction="ask.rule", payload={"source": 'ans("wrong")'})
    call("execute", instruction="rate " + wrong["receipt"], payload={"expected_answers": ['ans("8080")'],
         "reference": "independent fixture", "question": "Which port does the API use?"})
    call("teach", receipt=wrong["receipt"])
    built = dataset.build()
    assert any(r["meta"].get("source") == "machine.reference" and not r["label"] for r in built["kto"])
    assert any(r["meta"].get("validation") == "reference" and PROGRAM in r["chosen"] and 'ans("wrong")' in r["rejected"] for r in built["dpo"])


def test_teach_never_publishes_imported_private_objects_or_private_context(gd, tmp_path, monkeypatch):
    """Review 14.11.2026 (Claude): the owner sees its own private objects, import copies them into the machine, and teach used to
    publish the whole frozen execution (input objects, world, instance context) as internal — private knowledge laundered into the
    training data. Teach refuses with the reason; the owner can drop the object from the world (or not import it) and teach again."""
    from vi_mcp import dataset, evaluation
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path / "eval")
    g.new_object("erkenntnis", "own private API", ["service:api", "port:7777"], {"quelle": "fixture"}, visibility="privat", instance="a")
    call("create")
    call("import", filters=["service:api"])
    ex = run()
    assert ex["answers"] == ['ans("7777")']
    call("execute", instruction="rate " + ex["receipt"], payload={"expected_answers": ['ans("7777")'],
         "reference": "independent fixture", "question": "Which port does the API use?"})
    with pytest.raises(tools.Rejected, match="private"):
        call("teach", receipt=ex["receipt"])
    assert "7777" not in json.dumps(dataset.build())
    # Private instance context travels only with an explicit include_context.
    call("create", name="ctx")
    fact(name="ctx")
    call("context", name="ctx", payload={"messages": [{"role": "user", "content": "private note 4242"}]})
    ex2 = run(name="ctx")
    call("execute", name="ctx", instruction="rate " + ex2["receipt"], payload={"expected_answers": ['ans("80")'],
         "reference": "independent fixture", "question": "Which port does the API use?"})
    with pytest.raises(tools.Rejected, match="context"):
        call("teach", name="ctx", receipt=ex2["receipt"])
    assert call("teach", name="ctx", receipt=ex2["receipt"], include_context=True)["state"] == "taught"


def test_cli_passes_vil_and_payload_without_opening_the_server_database(gd, monkeypatch, tmp_path, capsys):
    from clingo_object import Memory
    seen = []
    monkeypatch.setattr(Memory, "call", lambda self, name, args: seen.append((name, args)) or '{"state":"stored"}')
    payload = tmp_path / "fact.json"
    payload.write_text(json.dumps({"source": "fixture", "hashtags": ["port:80"]}))
    assert machine.main(["execute", "test", "--instruction", "tell.fact api", "--payload", str(payload)]) == 0
    assert seen[0][0] == "vi_machine" and seen[0][1]["payload"]["hashtags"] == ["port:80"]


@pytest.mark.parametrize("field", ["weights_sha256", "tokenizer_sha256", "config_sha256", "decoder_sha256", "key_function"])
def test_server_frontend_refuses_every_unattested_pinned_identity_before_commit(gd, monkeypatch, field):
    from clingo_object.runtime import default_runtime
    call("create")
    fact()
    runtime = default_runtime()
    runtime["model"].update(model="test-model", **{field: "last-layer-v1" if field == "key_function" else "a" * 64})
    call("configure", payload=runtime)
    before = call()
    def llm(messages, info):
        info.update(model="test-model", knn={})
        return PROGRAM
    monkeypatch.setattr(logic, "_server_llm", llm)
    with pytest.raises(tools.Rejected, match="identity differs"):
        call("answer", question="Which port does the API use?")
    assert call() == before
