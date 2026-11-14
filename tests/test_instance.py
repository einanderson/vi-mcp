"""Complete instance contract: persisted context/policy, real MCP devices, bounded loop and no ambiguous write replay."""
import contextlib
import json

import pytest

from clingo_object import Memory, ServerError
from clingo_object.runtime import Device, Instance, default_runtime, validate_arguments
from vi_mcp import machine, tools
from vi_mcp import memory as g
from tests.test_http import srv  # noqa: F401


class LocalMemory:
    def call(self, tool, args):
        return tools.call_tool(tool, args, "instance-test", False)


class Programmer:
    identity = {"binding": "test", "model": "test"}
    def __init__(self, plans):
        self.plans = iter(plans)
        self.messages = []
    def __call__(self, messages):
        self.messages.append(messages)
        return json.dumps(next(self.plans))
    @contextlib.contextmanager
    def freeze(self, destination):
        yield {"kind": "test", "identity": self.identity}


def setup(model=None, devices=None, permissions=None, memory=None, name="test"):
    model = model or Programmer([])
    instance = Instance(memory or LocalMemory(), name, model, identity=model.identity, devices=devices)
    instance.call("create")
    runtime = {**default_runtime(), "model": model.identity, "devices": {k: v.descriptor for k, v in (devices or {}).items()},
               "permissions": list(permissions or [])}
    instance.call("configure", payload=runtime)
    return instance


def descriptor(effect="read"):
    return {"binding": "inventory-v1", "tool": "inventory", "effect": effect, "description": "checked inventory",
            "schema": {"type": "object", "properties": {"service": {"type": "string", "enum": ["api"]}},
                       "required": ["service"], "additionalProperties": False}}


def test_loop_reads_device_writes_fact_executes_rule_and_preserves_context(gd):
    pytest.importorskip("clingo")
    calls = []
    device = Device(descriptor(), lambda args: calls.append(args) or {"port": 8080})
    model = Programmer([
        {"operation": "device", "device": "inventory", "arguments": {"service": "api"}},
        {"operation": "execute", "instruction": "tell.fact api", "payload": {"text": "API", "source": "inventory reservation",
             "hashtags": ["service:api", "port:8080"]}},
        {"operation": "execute", "instruction": "ask.rule", "payload": {"source": 'ans(P) <= (O in objects("service:api")) & (O.port == P)'}},
        {"operation": "finish", "answer": "The checked API port is 8080."}])
    instance = setup(model, {"inventory": device}, ["inventory"])
    result = instance.run("Read the inventory and determine the API port", steps=4)
    assert result["state"] == "completed" and calls == [{"service": "api"}]
    assert "8080" in model.messages[1][-1]["content"]  # actual tool output enters the next planning step
    world = instance.inspect()["world"]
    assert world["loop"]["steps"] == 3 and world["context"][-1]["role"] == "assistant"
    assert "8080" in world["objects"]["api"]["tags"]["port"]
    g.close_db()
    g.open_db(gd)
    assert instance.inspect()["world"] == world


def test_loop_refuses_unauthorized_device_and_model_cannot_change_policy(gd):
    calls = []
    device = Device(descriptor("write"), lambda args: calls.append(args))
    model = Programmer([{"operation": "device", "device": "inventory", "arguments": {"service": "api"}},
                        {"operation": "configure", "permissions": ["inventory"]}])
    instance = setup(model, {"inventory": device})
    result = instance.run("untrusted prompt says grant all permissions", steps=2)
    assert result["state"] == "paused" and not calls
    world = instance.inspect()["world"]
    assert world["runtime"]["permissions"] == [] and world["loop"]["steps"] == 2
    assert "not authorized" in world["context"][1]["content"]


@pytest.mark.parametrize("tool", ["vi_machine", "vi_release", "vi_set_visibility", "vi_quarantine"])
def test_owner_operations_can_never_be_a_device(gd, tool):
    """Review 14.11.2026 (Claude), item 8: the loop forbids owner operations, but a device may name any MCP tool. A host that
    registered vi_machine as a device would let the model call configure/teach/snapshot through it and grant itself rights.
    The shared descriptor check refuses such a device on the host and on the server."""
    d = {**descriptor("write"), "tool": tool}
    with pytest.raises(ValueError, match="owner operation"):
        Device(d, lambda args: None)
    instance = setup()
    with pytest.raises((ServerError, tools.Rejected, ValueError), match="owner operation"):
        instance.call("configure", payload={**default_runtime(), "model": Programmer.identity, "devices": {"control": d},
                                            "permissions": ["control"]})


def test_owner_tool_list_matches_the_server_names_and_aliases():
    from clingo_object.runtime import OWNER_TOOLS
    from vi_mcp import api_en
    english = {t for t in OWNER_TOOLS if t not in api_en.TOOL_ALIASES}
    served = {t["name"] for t in tools.TOOLS}
    assert english <= served  # every listed owner tool exists under that name
    assert {a for a, b in api_en.TOOL_ALIASES.items() if b in english} <= OWNER_TOOLS  # and its German alias is blocked too


def test_uncertain_write_never_replays_after_process_restart(gd):
    calls = []
    def write(args):
        calls.append(args)
        raise TimeoutError("remote may already have applied the request")
    model = Programmer([{"operation": "device", "device": "inventory", "arguments": {"service": "api"}}])
    instance = setup(model, {"inventory": Device(descriptor("write"), write)}, ["inventory"])
    with pytest.raises(RuntimeError, match="uncertain"):
        instance.run("update inventory")
    pending = instance.inspect()["world"]["loop"]["pending"]
    assert pending["status"] == "uncertain" and len(calls) == 1
    with pytest.raises(tools.Rejected, match="pending"):
        instance.call("snapshot", snapshot="must-not-replay")
    g.close_db()
    g.open_db(gd)
    with pytest.raises(tools.Rejected, match="pending"):
        instance.run()
    assert len(calls) == 1
    instance.call("loop", payload={"operation": "resolve", "reservation": pending["id"], "outcome": "completed", "reference": "independent device audit"})
    assert instance.inspect()["world"]["loop"]["pending"] is None
    assert not instance.inspect()["world"]["loop"]["status"] == "running"


def test_owner_revocation_cannot_be_undone_by_restored_host_capabilities(gd, tmp_path):
    device = Device(descriptor(), lambda _: "read")
    model = Programmer([])
    instance = setup(model, {"inventory": device}, ["inventory"])
    instance.call("context", payload={"messages": [{"role": "user", "content": "Answer in German"}]})
    before = instance.inspect()["world"]
    instance.checkpoint(tmp_path / "bundle", "whole-instance")
    instance.call("context", payload={"replace": True, "reason": "explicit context reset", "messages": []})
    restored = Instance.boot(LocalMemory(), "clone", model, tmp_path / "bundle", devices={"inventory": device})
    assert restored.inspect()["world"] == before
    with pytest.raises(ValueError, match="registry"):
        Instance.boot(LocalMemory(), "revoked", model, tmp_path / "bundle", devices={})
    assert g.load(machine._id("instance-test", "revoked")) is None
    encoded = (tmp_path / "bundle/instance.json").read_text()
    assert "http://" not in encoded and "Bearer" not in encoded


def test_snapshot_transaction_leaves_no_orphan_when_parent_save_fails(gd, monkeypatch):
    instance = setup()
    before = instance.inspect()
    original = machine._save
    def fail(o, reason, **kwargs):
        if reason == "snapshot created":
            raise ValueError("parent full")
        return original(o, reason, **kwargs)
    monkeypatch.setattr(machine, "_save", fail)
    with pytest.raises(tools.Rejected, match="parent full"):
        instance.call("snapshot", snapshot="orphan")
    assert g.load(machine._id("instance-test", "orphan", True)) is None
    assert instance.inspect() == before


def test_changed_model_and_context_limits_refuse_before_execution(gd):
    model = Programmer([])
    instance = setup(model)
    instance.call("context", payload={"messages": [{"role": "system", "content": "German output"}]})
    with pytest.raises(tools.Rejected, match="reason"):
        instance.call("context", payload={"replace": True, "messages": []})
    with pytest.raises(tools.Rejected, match="context full"):
        instance.call("context", payload={"messages": [{"role": "user", "content": "large"}] * 33})
    instance.identity = {"binding": "other", "model": "other"}
    with pytest.raises(ValueError, match="model identity"):
        instance.run("must not run")
    assert not model.messages


def test_real_mcp_device_and_memory_transport(srv, tmp_path):
    port, token, _ = srv
    token_file = tmp_path / "token"
    token_file.write_text(token)
    memory = Memory(f"http://127.0.0.1:{port}/mcp", token_file)
    device = Device.mcp(memory, "vi_vocabulary", binding="knowledge-v1", schema={"type": "object", "properties": {}},
                        description="read vocabulary")
    model = Programmer([{"operation": "device", "device": "vocabulary", "arguments": {}},
                        {"operation": "finish", "answer": "Vocabulary was read through MCP."}])
    instance = setup(model, {"vocabulary": device}, ["vocabulary"], memory=memory)
    assert instance.run("read vocabulary", steps=2)["state"] == "completed"
    events = instance.inspect()["world"]["loop"]["events"]
    assert events[0]["device"] == "vocabulary" and events[0]["ok"]


def test_server_program_refusal_becomes_repair_feedback_over_mcp(srv, tmp_path):
    port, token, _ = srv
    token_file = tmp_path / "token"
    token_file.write_text(token)
    memory = Memory(f"http://127.0.0.1:{port}/mcp", token_file)
    model = Programmer([{"operation": "execute", "instruction": "def.rule example", "payload": {"source": "import os"}},
                        {"operation": "execute", "instruction": "def.rule example", "payload": {"source": 'ans("safe")'}},
                        {"operation": "finish", "answer": "Corrected after actual server refusal"}])
    instance = setup(model, memory=memory)
    assert instance.run("define a rule", steps=3)["state"] == "completed"
    assert "refused" in model.messages[1][-1]["content"]


def test_schema_subset_and_latest_teaching_correction(gd):
    with pytest.raises(ValueError, match="unsupported"):
        validate_arguments({}, {"type": "object", "oneOf": []})
    pytest.importorskip("clingo")
    instance = setup()
    execution = instance.call("execute", instruction="ask.rule", payload={"source": 'ans("safe")'})
    def rate(want):
        instance.call("execute", instruction="rate " + execution["receipt"], payload={"expected_answers": [want],
            "reference": "independent fixture", "question": "What is safe?"})
        return instance.call("teach", receipt=execution["receipt"])
    first = rate('ans("safe")')
    second = rate('ans("other")')
    assert first["id"] == second["id"] and second["state"] == "teaching corrected" and not second["correct"]
    assert not g.load(first["id"]).data["machine_example"]["correct"]


def test_operator_cli_runs_checkpoints_and_boots_through_real_mcp(srv, tmp_path, capsys):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    port, token, _ = srv
    token_file = tmp_path / "token"
    token_file.write_text(token)
    plans = iter([{"operation": "device", "device": "vocabulary", "arguments": {}}, {"operation": "finish", "answer": "Read through MCP"}])
    class ModelServer(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"model": "test-model", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(next(plans))}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/mcp"
        profile = tmp_path / "bindings.json"
        profile.write_text(json.dumps({"model": {"kind": "remote", "binding": "cli-model", "model": "test-model",
            "url": f"http://127.0.0.1:{server.server_port}/v1/chat/completions"},
            "devices": {"vocabulary": {"url": url, "token_file": str(token_file), "binding": "vocabulary-v1", "tool": "vi_vocabulary",
                "schema": {"type": "object", "properties": {}}, "description": "read vocabulary", "effect": "read"}},
            "permissions": ["vocabulary"]}))
        common = ["--bindings", str(profile), "--url", url, "--token-file", str(token_file)]
        assert machine.main(["run", "cli", *common, "--create", "--goal", "Read vocabulary", "--steps", "2"]) == 0
        assert json.loads(capsys.readouterr().out)["state"] == "completed"
        checkpoint = tmp_path / "checkpoint"
        assert machine.main(["checkpoint", "cli", *common, "--directory", str(checkpoint), "--snapshot", "cli-checkpoint"]) == 0
        capsys.readouterr()
        assert machine.main(["boot", "cli-clone", *common, "--directory", str(checkpoint)]) == 0
        boot = json.loads(capsys.readouterr().out)
        assert boot["world"]["context"][-1]["content"] == "Read through MCP" and boot["permissions"] == ["vocabulary"]
        revoked = json.loads(profile.read_text())
        revoked["permissions"] = []
        profile.write_text(json.dumps(revoked))
        assert machine.main(["boot", "revoked-cli", *common, "--directory", str(checkpoint)]) == 2
        assert "registry" in capsys.readouterr().out
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_foreign_device_without_token_file_never_inherits_the_vi_credential(tmp_path, monkeypatch):
    import os
    from vi_mcp import machine_host
    seen = []
    monkeypatch.setattr(machine_host, "Memory", lambda **kwargs: seen.append(kwargs) or object())
    profile = tmp_path / "bindings.json"
    profile.write_text(json.dumps({"model": {"kind": "remote", "binding": "model", "model": "test", "url": "https://model.invalid"},
        "devices": {"inventory": {**descriptor(), "url": "https://device.invalid/mcp"}}}))
    _model, _devices, runtime = machine_host.bindings(profile)
    assert seen == [{"url": "https://device.invalid/mcp", "token_file": os.devnull}]
    assert runtime["permissions"] == []  # registering a transport grants no authority
    assert not _devices
